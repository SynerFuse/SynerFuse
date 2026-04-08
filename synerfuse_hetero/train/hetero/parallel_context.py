import os
import math
import warnings
import itertools
import operator
from typing import List, Optional
from datetime import timedelta
from functools import cmp_to_key
from collections import defaultdict

import torch

def get_nccl_options(pg_name, nccl_comm_cfgs):
    from synerfuse.core.parallel_state import get_nccl_options 
    return get_nccl_options(pg_name, nccl_comm_cfgs)

def get_group_name(token, independent_ep=False):
    names = {
        "tp-ep-pp": ("mp_exp", None),  # (with_ep, without_ep)
        "tp-ep": ("tp_exp", None),
        "ep": ("exp", None),
        "dp": ("dp_modulo_exp", "dp"),
        "dp-cp": ("dp_cp", "dp_modulo_exp_cp"), 
        "cp": ("cp", "cp"),
        "tp-pp": ("mp", "mp"),
        "tp": ("tp", "tp"),
        "pp": ("pp", "pp"),
        "tp-dp-cp": ("tp_dp_cp", "tp_dp_cp"),
        "tp-dp": ("tp_dp", "tp_dp"),
        "tp-cp": (None, "tp_cp")
        # "usp": ("usp", "usp"),
    }
    name_pair = names.get(token, None)
    if name_pair is None:
        raise ValueError(f"Invalid token: {token}")
    if independent_ep:
        name = name_pair[0]
        assert name is not None, f"Token {token} does not support independent ep."
        return name
    else:
        name = name_pair[1]
        assert name is not None, f"Token {token} does not support non-independent ep."
        return name

def find_overlapped_mapping(dim1, dim2, global_size=None):
    """
    Finds the overlapped mapping between two dimensions within an optional global size. Please refer to https://eli.thegreenplace.net/2008/08/15/intersection-of-1d-segments for details.
    """
    # Calculate the least common multiple (LCM) of dim1 and dim2, or use global_size if provided
    dim_lcm = global_size if global_size else math.lcm(dim1, dim2)
    # Generate segments for dim1 and dim2
    dim1_segments_len = dim_lcm // dim1
    dim1_segments = [(i * dim1_segments_len, (i + 1) * dim1_segments_len) for i in range(dim1)]
    dim2_segments_len = dim_lcm // dim2
    dim2_segments = [(i * dim2_segments_len, (i + 1) * dim2_segments_len) for i in range(dim2)]
    # Initialize the mapping of overlapped segments
    overlapped_mapping = {i: [] for i in range(dim1)}
    # Calculate overlaps between dim1 and dim2 segments
    for i, (start1, end1) in enumerate(dim1_segments):
        for j, (start2, end2) in enumerate(dim2_segments):
            if start1 < end2 and end1 > start2:  # Check if segments overlap
                # Calculate the overlap offsets relative to the start of the dim1 segment
                local_overlap_start1 = max(start1, start2) - start1
                local_overlap_end1 = min(end1, end2) - start1
                overlapped_mapping[i].append(
                    (j, local_overlap_start1, local_overlap_end1)
                )
    return overlapped_mapping


def get_segment_indices(length, n, i):
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if i < 0 or i >= n:
        raise ValueError("i must be in the range [0, n-1]")

    base = length // n
    remainder = length % n

    if i < remainder:
        start = i * (base + 1)
        end = start + (base + 1)
    else:
        start = remainder * (base + 1) + (i - remainder) * base
        end = start + base

    return start, end


def find_overlapped_mapping_tp_cp(sequence_mesh1, sequence_mesh2, global_size=None, sequence_parallel=True, cp_size1=2, tp_size1=None, tp_size2=None):
    overlapped_mapping = {}
    tp_overlapped_mapping = None
    if not sequence_parallel and tp_size1 is not None and tp_size2 is not None:
        tp_overlapped_mapping = find_overlapped_mapping(tp_size1, tp_size2)
        
    for src_sp_dim, src_data in sequence_mesh1.items():
        overlapped_mapping[src_sp_dim] = []

        elem_to_src_sp_dim = defaultdict(list)
        for index, value in enumerate(src_data):
            elem_to_src_sp_dim[value].append(index)    

        for dst_sp_dim, dst_data in sequence_mesh2.items():
            if tp_overlapped_mapping is not None:
                src_tp = src_sp_dim % tp_size1
                dst_tp = dst_sp_dim % tp_size2
                valid_dst_tps = [dim for dim, _, _ in tp_overlapped_mapping[src_tp]]
                if dst_tp not in valid_dst_tps:
                    continue

            common_elements = list(set(src_data) & set(dst_data))
            if common_elements:
                common_elements.sort()
                cur_overlap_start_end_lst = []
                for elem in common_elements:
                    cur_index = elem_to_src_sp_dim[elem][0]
                    if sequence_parallel:
                        local_overlap_start, local_overlap_end = get_segment_indices(
                            global_size//len(sequence_mesh1.keys()), len(src_data), cur_index)
                    else:
                        local_overlap_start, local_overlap_end = get_segment_indices(
                            global_size // cp_size1, len(src_data), cur_index)
                    cur_overlap_start_end_lst.append([local_overlap_start, local_overlap_end])
                # 芍?D?????o?2⊿
                # merged_overlap_start_end_lst = [cur_overlap_start_end_lst[0]]
                # for current in cur_overlap_start_end_lst:
                #     last = merged_overlap_start_end_lst[-1]
                #     if last[1] == current[0]:
                #         last[1] = current[1]
                #     else:
                #         merged_overlap_start_end_lst.append(current)
                overlapped_mapping[src_sp_dim].append(
                    (dst_sp_dim, cur_overlap_start_end_lst)
                )
    return overlapped_mapping
                    

class RankMapper:
    def __init__(self, args):
        assert (
            torch.distributed.is_initialized()
        ), "torch.distributed is not initialized"
        self._world_size = torch.distributed.get_world_size()
        # The order of device types is very import for creating the logical rank.
        # Users should make sure the order satisfies their needs. 
        self._hetero_device_types = args.hetero_device_types
        self._hetero_current_device_type = args.hetero_current_device_type
        self._rank_infos = {}
        self._physical_rank_to_logical_rank = {}
        self._logical_rank_to_physical_rank = {}
        self.build_rank_mapping()

    def build_rank_mapping(self):
        # Collect all rank infos.
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        all_rank_infos = [None] * world_size
        cur_rank_info = {'rank': rank,
                         'device_type': self._hetero_current_device_type}
        torch.distributed.all_gather_object(
            all_rank_infos, cur_rank_info)

        physical_ranks = []
        for info in all_rank_infos:
            self._rank_infos[info['rank']] = info
            physical_ranks.append(info['rank'])

        # Sort the physical ranks by device type and rank.
        def _compare(rank1, rank2):
            device_type1 = self._rank_infos[rank1]['device_type']
            device_type2 = self._rank_infos[rank2]['device_type']
            if self._hetero_device_types \
                and self._hetero_device_types.index(device_type1) < self._hetero_device_types.index(device_type2):
                return -1
            elif self._hetero_device_types \
                and self._hetero_device_types.index(device_type1) > self._hetero_device_types.index(device_type2):
                return 1
            else:
                return rank1 - rank2
        sorted_physical_ranks = sorted(
            physical_ranks, key=cmp_to_key(_compare))

        # Build the mapping between physical rank and logical rank
        for logical_rank, physical_rank in enumerate(sorted_physical_ranks):
            self._physical_rank_to_logical_rank[physical_rank] = logical_rank 
            self._logical_rank_to_physical_rank[logical_rank] = physical_rank

    def to_physical_ranks(self, logical_ranks: list) -> list:
        """Converts logical ranks to physical ranks."""
        physical_ranks = []
        for logical_rank in logical_ranks:
            physical_ranks.append(
                self._logical_rank_to_physical_rank[logical_rank])
        return physical_ranks

    def to_logical_ranks(self, physical_ranks: list) -> list:
        """Converts physical ranks to logical ranks."""
        logical_ranks = []
        for physical_rank in physical_ranks:
            logical_ranks.append(
                self._physical_rank_to_logical_rank[physical_rank])
        return logical_ranks


class ProcessMesh:
    """ Define n-dimensional Cartesian process topology. """

    def __init__(
        self,
        data_parallel_size: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_split_rank: Optional[int] = None,
        use_sharp: bool = False,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        expert_tensor_parallel_size: Optional[int] = None,
        nccl_communicator_config_path: Optional[str] = None,
        distributed_timeout_minutes: int = 30,
        order: str = "tp-cp-ep-dp-pp",
        offset: int = 0,
        rank_mapper: RankMapper = None,
    ):
        assert torch.distributed.is_initialized()
        self._data_parallel_size = data_parallel_size
        self._tensor_model_parallel_size = tensor_model_parallel_size
        self._pipeline_model_parallel_size = pipeline_model_parallel_size
        self._virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self._pipeline_model_parallel_split_rank = pipeline_model_parallel_split_rank
        self._use_sharp = use_sharp
        self._context_parallel_size = context_parallel_size
        self._expert_model_parallel_size = expert_model_parallel_size
        self._expert_tensor_parallel_size = expert_tensor_parallel_size
        self._order = order
        self._offset = offset

        if data_parallel_size % expert_model_parallel_size != 0:
            raise RuntimeError(
                f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
            )

        if expert_model_parallel_size > 1 and context_parallel_size > 1:
            raise RuntimeError(
                f"combination of expert model prallellism and context parallelism is not supported"
            )

        self._timeout = timedelta(minutes=distributed_timeout_minutes)
        self._rank = torch.distributed.get_rank()
        self._world_size = (
            self._tensor_model_parallel_size
            * self._pipeline_model_parallel_size
            * self._context_parallel_size
            * self._data_parallel_size
        )

        if self._virtual_pipeline_model_parallel_size is not None:
            if not self._pipeline_model_parallel_size > 2:
                raise RuntimeError(
                    "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
                )

            self._virtual_pipeline_model_parallel_rank = 0
            self._virtual_pipeline_model_parallel_world_size = self._virtual_pipeline_model_parallel_size

        self._nccl_comm_cfgs = {}
        if nccl_communicator_config_path is not None:
            try:
                import yaml
            except ImportError:
                raise RuntimeError(
                    "Cannot import `yaml`. Setting custom nccl communicator configs "
                    "requires the yaml package."
                )

            with open(nccl_communicator_config_path, "r") as stream:
                self._nccl_comm_cfgs = yaml.safe_load(stream)

        from synerfuse.core.parallel_state import RankGenerator 
        self._rank_generator = RankGenerator(
            tp=tensor_model_parallel_size,
            ep=expert_model_parallel_size,
            dp=data_parallel_size,
            pp=pipeline_model_parallel_size,
            cp=context_parallel_size,
            order=order,
        )

        self._timeout = timedelta(minutes=distributed_timeout_minutes)
        self._rank_mapper = rank_mapper 
        self._group_ranks = {} # group_ranks belongs to the current rank 
        self._all_group_ranks = defaultdict(list) # group_ranks belongs to the current process mesh 
        self._process_groups = {} # process groups belongs to the current rank
        self._process_groups_gloo = {} # process groups belongs to the current rank with gloo backend

        self.build_all_process_groups()

    def build_process_group(
        self, token, independent_ep=False, gloo=False
    ):
        logical_ranks_list = self._rank_generator.get_ranks(token, independent_ep=independent_ep)
        # Add the offset for each ranks of the current process mesh
        for logical_ranks in logical_ranks_list:
            for i in range(len(logical_ranks)):
                logical_ranks[i] += self._offset 

        for logical_ranks in logical_ranks_list:
            group_name = get_group_name(token, independent_ep=independent_ep) 
            pg_options = get_nccl_options(group_name, self._nccl_comm_cfgs)
            ranks = self._rank_mapper.to_physical_ranks(logical_ranks)
            group = torch.distributed.new_group(
                ranks,
                timeout=self._timeout,
                pg_options=pg_options,
            )
            if gloo:
                group_gloo = torch.distributed.new_group(
                    ranks, timeout=self._timeout, backend="gloo"
                )
            self._all_group_ranks[group_name].append(ranks)
            if self._rank in ranks:
                self._group_ranks[group_name] = ranks
                self._process_groups[group_name] = group
                if gloo:
                    self._process_groups_gloo[group_name] = group_gloo

            # if token == "pp":
            #     if len(ranks) > 1:
            #         embedding_ranks = [ranks[0], ranks[-1]]
            #         position_embedding_ranks = [ranks[0]]
            #         if self._pipeline_model_parallel_split_rank is not None:
            #             if (
            #                 ranks[self._pipeline_model_parallel_split_rank]
            #                 not in embedding_ranks
            #             ):
            #                 embedding_ranks = [
            #                     ranks[0],
            #                     ranks[self._pipeline_model_parallel_split_rank],
            #                     ranks[-1],
            #                 ]
            #             if (
            #                 ranks[self._pipeline_model_parallel_split_rank]
            #                 not in position_embedding_ranks
            #             ):
            #                 position_embedding_ranks = [
            #                     ranks[0],
            #                     ranks[self._pipeline_model_parallel_split_rank],
            #                 ]
            #     else:
            #         embedding_ranks = ranks
            #         position_embedding_ranks = ranks

            #     self._all_group_ranks["embd"].append(embedding_ranks)
            #     group = torch.distributed.new_group(
            #         embedding_ranks,
            #         timeout=self._timeout,
            #         pg_options=get_nccl_options("embd", self._nccl_comm_cfgs),
            #     )
            #     if self._rank in embedding_ranks:
            #         self._process_groups["embd"] = group
            #     # TODO: need to check whether self._rank in ranks is correct
            #     # or should it be self._rank in embedding_ranks
            #     if self._rank in ranks:
            #         self._group_ranks["embd"] = embedding_ranks

            #     self._all_group_ranks["embd_pos"].append(position_embedding_ranks)
            #     group = torch.distributed.new_group(
            #         position_embedding_ranks,
            #         timeout=self.timeout,
            #         pg_options=get_nccl_options("embd", self._nccl_comm_cfgs),
            #     )
            #     if self._rank in position_embedding_ranks:
            #         self._process_groups["embd_pos"] = group
            #     # TODO: need to check whether self._rank in ranks is correct
            #     # or should it be self._rank in position_embedding_ranks
            #     if self._rank in ranks:
            #         self._group_ranks["embd_pos"] = position_embedding_ranks

        # if token == "pp":
        #     self._last_rank_when_using_pp = self._rank_mapper.to_physical_ranks(
        #         [logical_ranks_list[-1][-1]]
        #     )[0]

    def build_all_process_groups(self):
        self.build_process_group("dp", independent_ep=False, gloo=True)
        self.build_process_group("dp-cp", independent_ep=False, gloo=True)

        # Apply SHARP to DP process groups
        if self._use_sharp:
            if self._rank == 0:
                print(
                    "The number of process groups to use SHARP with depends on the type "
                    "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                    "process groups and QM2 supports up to 256 process groups. We apply "
                    "SHARP to the communications of the data-parallel domain. If the "
                    "number of data-parallel process groups is larger than the max "
                    "process groups that the network switch supports, the communication "
                    "will fall back to non-SHARP operators. To enable SHARP, "
                    "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
                )
            torch.distributed.barrier(
                group=self.get_process_group("dp-cp"),
                device_ids=[torch.cuda.current_device()],
            )
            # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
            os.environ["NCCL_COLLNET_ENABLE"] = "0"

        self.build_process_group("cp", independent_ep=False, gloo=False)
        self.build_process_group("tp-pp", independent_ep=False, gloo=False)
        self.build_process_group("tp-ep-pp", independent_ep=True, gloo=False)
        self.build_process_group('tp', independent_ep=False, gloo=False)
        self.build_process_group("pp", independent_ep=False, gloo=False)
        self.build_process_group("tp-dp-cp", independent_ep=False, gloo=False)
        self.build_process_group("tp-dp", independent_ep=False, gloo=False)
        self.build_process_group("tp-cp", independent_ep=False, gloo=False)
        self.build_process_group("tp-ep", independent_ep=True, gloo=False)
        self.build_process_group("ep", independent_ep=True, gloo=False)
        self.build_process_group("dp", independent_ep=True, gloo=True)
        self.build_process_group("dp-cp", independent_ep=False, gloo=True)
        self.build_process_group("dp-cp", independent_ep=True, gloo=True)
        # self.build_process_group("usp", independent_ep=True, gloo=True)

    def get_parallel_size(self, token, independent_ep=False):
        if independent_ep:
            parallel_sizes = self._rank_generator.ordered_size_w_ep
            order = self._rank_generator.order_w_ep
        else:
            parallel_sizes = self._rank_generator.ordered_size_wo_ep
            order = self._rank_generator.order_wo_ep
        order = order.split("-")
        if token in order:
            return parallel_sizes[order.index(token)]
        else:
            raise ValueError(f"Invalid token: {token}")

    

    def get_process_group(
        self,
        token,
        independent_ep=False,
        gloo=False,
        check_initialized=False,
    ):
        group_name = get_group_name(token, independent_ep=independent_ep)
        if gloo:
            group = self._process_groups_gloo.get(group_name, None)
        else:
            group = self._process_groups.get(group_name, None)
        if check_initialized:
            assert (
                group is not None
            ), f"Process group {group_name} is not initialized."
        return group

    def get_process_group_size(self, token, independent_ep=False, gloo=False):
        group_name = get_group_name(token, independent_ep=independent_ep)
        if gloo:
            return torch.distributed.get_world_size(self._process_groups_gloo[group_name])
        else:
            return torch.distributed.get_world_size(self._process_groups[group_name])

    def get_process_group_ranks(
        self, token, independent_ep=False, check_initialized=False
    ):
        group_name = get_group_name(token, independent_ep=independent_ep)
        ranks = self._group_ranks.get(group_name, None)
        if check_initialized:
            assert (
                ranks is not None
            ), f"Process group {group_name} is not initialized."
        return ranks

    def get_all_process_group_ranks(
        self, token, independent_ep=False, check_initialized=False
    ):
        group_name = get_group_name(token, independent_ep=independent_ep)
        ranks = self._all_group_ranks.get(group_name, None)
        if check_initialized:
            assert (
                ranks is not None
            ), f"Process group {group_name} is not initialized."
        return ranks

    def logical_coords_to_physical_ranks(self, coords, independent_ep=False):
        def _prefix_product(a: List[int], init=1) -> List[int]:
            r = [init]
            for v in a:
                init = init * v
                r.append(init)
            return r
        if independent_ep:
            for coord in coords:  
                assert len(coord) == 5
            sizes = self._rank_generator.ordered_size_w_ep
        else:
            for coord in coords:  
                assert len(coord) == 4
            sizes = self._rank_generator.ordered_size_wo_ep
        strides = _prefix_product(sizes)
        logical_ranks = []
        for coord in coords:  
            logical_rank = sum([c * s for c, s in zip(coord, strides)]) + self._offset
            logical_ranks.append(logical_rank)
        ranks = self._rank_mapper.to_physical_ranks(logical_ranks)
        return ranks


class ParallelContext:
    def __init__(self, args):
        assert torch.distributed.is_initialized()
        self._is_initialized = False 
        self._args = args
        self._current_process_mesh_index = 0 
        self._process_meshes = [] 
        self._rank_to_process_mesh = {}
        self._inter_mesh_group_ranks = defaultdict(list) 
        # self._inter_mesh_process_groups = {} # (src_rank, dst_rank) -> group
        self._inter_mesh_process_groups_pp = {} # (src_rank, dst_rank) -> bool
        self._inter_mesh_process_groups_dp = {} # (src_rank, dst_rank) -> bool
        self._inter_mesh_process_groups_edp = {} # (src_rank, dst_rank) -> bool
        # (src_rank, local_tensor_shape, next) -> (dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size)
        self._inter_mesh_tensor_slices = {}
        self._group_ranks = defaultdict(list)
        self._all_group_ranks = defaultdict(list)
        self._process_groups = defaultdict(list) 
        self._process_group_to_ranks = {}
        self._parallel_world_sizes = {}
        self._parallel_ranks = {}
        self._timeout = timedelta(minutes=self._args.distributed_timeout_minutes)

        self._rank = torch.distributed.get_rank()
        self._rank_mapper = RankMapper(args)
        self.build_all_process_meshes()
        self.build_all_inter_mesh_process_groups()
        self.build_global_process_groups()
        from synerfuse.core.utils import GlobalMemoryBuffer
        self._global_memory_buffer = GlobalMemoryBuffer()

        self._is_initialized = True

    def is_initialized(self):
        return self._is_initialized

    def build_all_process_meshes(self):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        logical_rank = self._rank_mapper.to_logical_ranks([rank])[0]
        accumulated_world_size = 0
        for tp, cp, ep, dp, pp in self._args.hetero_process_meshes:
            process_mesh = ProcessMesh(
                tensor_model_parallel_size=tp,
                context_parallel_size=cp,
                data_parallel_size=dp,
                pipeline_model_parallel_size=pp,
                expert_model_parallel_size=ep,
                nccl_communicator_config_path=self._args.nccl_communicator_config_path,
                distributed_timeout_minutes=self._args.distributed_timeout_minutes,
                order='tp-cp-ep-dp-pp' if not self._args.use_tp_pp_dp_mapping else 'tp-pp-dp',
                offset=accumulated_world_size,
                rank_mapper=self._rank_mapper,
            )
            if (
                logical_rank >= accumulated_world_size
                and logical_rank < accumulated_world_size + process_mesh._world_size
            ):
                self._current_process_mesh_index = len(self._process_meshes)
            accumulated_world_size += process_mesh._world_size
            self._process_meshes.append(process_mesh)
        if world_size != accumulated_world_size:
            raise RuntimeError(
                f"World size mismatch. Expected {world_size}, but got {accumulated_world_size}"
            )

        all_rank_to_process_mesh = [None for _ in range(world_size)]
        cur_rank_to_process_mesh = {
            "rank": rank,
            "process_mesh_idx": self._current_process_mesh_index,
        }
        torch.distributed.all_gather_object(
            all_rank_to_process_mesh, cur_rank_to_process_mesh
        )
        for item in all_rank_to_process_mesh:
            self._rank_to_process_mesh[item["rank"]] = self._process_meshes[
                item["process_mesh_idx"]
            ]

    
    def distribute_data(self, tp_size, cp_size, chunks, sequence_parallel=True):
        lst = list(range(1, chunks + 1))

        cp_blocks = []
        block_size = chunks // (2 * cp_size)
        for i in range(2 * cp_size):
            start = i * block_size
            end = start + block_size
            cp_blocks.append(lst[start:end])

        merged_cp_data = []
        for cp_rank in range(cp_size):
            idx1 = cp_rank
            idx2 = 2 * cp_size - 1 - cp_rank
            merged = cp_blocks[idx1] + cp_blocks[idx2]
            merged_cp_data.append(merged)

        allocation = {}

        if sequence_parallel:
            for cp_rank in range(cp_size):
                cp_block = merged_cp_data[cp_rank]
                tp_group_size = len(cp_block) // tp_size
                for tp_rank in range(tp_size):
                    start = tp_rank * tp_group_size
                    end = start + tp_group_size
                    sp_dim = tp_rank * 1 + cp_rank * tp_size
                    allocation[sp_dim] = cp_block[start:end]
        else:
            for cp_rank in range(cp_size):
                cp_block = merged_cp_data[cp_rank]
                for tp_rank in range(tp_size):
                    sp_dim = tp_rank * 1 + cp_rank * tp_size
                    allocation[sp_dim] = cp_block

        return allocation

    def build_inter_mesh_process_groups_tp_cp(self, process_mesh1, process_mesh2):
        tp1 = process_mesh1.get_parallel_size("tp", independent_ep=False)
        cp1 = process_mesh1.get_parallel_size("cp", independent_ep=False)
        dp1 = process_mesh1.get_parallel_size("dp", independent_ep=False)
        tp2 = process_mesh2.get_parallel_size("tp", independent_ep=False)
        cp2 = process_mesh2.get_parallel_size("cp", independent_ep=False)
        dp2 = process_mesh2.get_parallel_size("dp", independent_ep=False)

        if self._args.sequence_parallel:
            chunks = math.lcm(tp1*cp1*2, tp2*cp2*2)
        else:
            chunks = math.lcm(cp1*2, cp2*2)
        # key: tp_rank + cp_rank * tp_size, value: ﹞???那y?Y 
        sequence_mesh1 = self.distribute_data(tp1, cp1, chunks, sequence_parallel=self._args.sequence_parallel)
        sequence_mesh2 = self.distribute_data(tp2, cp2, chunks, sequence_parallel=self._args.sequence_parallel)
        if self._args.sequence_parallel:
            sp_overlapped_mapping = find_overlapped_mapping_tp_cp(sequence_mesh1, sequence_mesh2, global_size=chunks, sequence_parallel=True)
        else:
            # without sequence_parallel
            sp_overlapped_mapping = find_overlapped_mapping_tp_cp(sequence_mesh1, sequence_mesh2, global_size=chunks, sequence_parallel=False, cp_size1=cp1, tp_size1=tp1, tp_size2=tp2)
        dp_overlapped_mapping = find_overlapped_mapping(dp1, dp2)
        tp_overlapped_mapping = None
        if not self._args.sequence_parallel:
            tp_overlapped_mapping = find_overlapped_mapping(tp1, tp2)
            
        src_pp_dims = [process_mesh1.get_parallel_size("pp") - 1]
        dst_pp_dims = [0]
        for src_sp_dim in sp_overlapped_mapping.keys():
            src_tp_rank, src_cp_rank = src_sp_dim % tp1, src_sp_dim // tp1
            for src_dp_dim in range(dp1):
                src_coord = [src_tp_rank, src_cp_rank, src_dp_dim, src_pp_dims[0]]
                
                valid_dst_sp_dims = []
                for dim, _ in sp_overlapped_mapping[src_sp_dim]:
                    if not self._args.sequence_parallel and tp_overlapped_mapping is not None:
                        src_tp_i = src_sp_dim % tp1
                        dst_tp_i = dim % tp2
                        valid_dst_tps = [t for t, _, _ in tp_overlapped_mapping[src_tp_i]]
                        if dst_tp_i not in valid_dst_tps:
                            continue
                    valid_dst_sp_dims.append(dim)

                dst_dp_dims = [c for c, _, _ in dp_overlapped_mapping[src_dp_dim]]
                dst_coords = list(
                    itertools.product(valid_dst_sp_dims, dst_dp_dims, dst_pp_dims)
                )
                src_rank = process_mesh1.logical_coords_to_physical_ranks(
                    [src_coord]
                )[0]
                
                for i, dst_coord in enumerate(dst_coords):
                    dst_sp_dim, dst_dp_dim, dst_pp_dim = dst_coord
                    dst_coord = [dst_sp_dim % tp2, dst_sp_dim // tp2, dst_dp_dim, dst_pp_dim]
                    dst_rank = process_mesh2.logical_coords_to_physical_ranks(
                        [dst_coord]
                    )[0]
                    # NOTE: There is no need to create a group for the commnetting boundary.
                    #       We will create the `pp` group in the `build_global_process_groups` function.
                    # ranks = [src_rank, dst_rank]
                    # timeout = max(process_mesh1._timeout, process_mesh2._timeout)
                    # group = torch.distributed.new_group(ranks, timeout=timeout)
                    self._inter_mesh_process_groups_pp[(src_rank, dst_rank)] = True
        # find mp(tp+pp) group connection
        for k in range(dp1):
            src_coord = [tp1 - 1, cp1 - 1, k, src_pp_dims[0]]
            dst_dp_dims = [dim for dim, _, _ in dp_overlapped_mapping[k]]
            dst_coords = list(
                itertools.product([0], [0], dst_dp_dims, dst_pp_dims)
            )
            src_rank = process_mesh1.logical_coords_to_physical_ranks(
                [src_coord]
            )[0]
            for dst_coord in dst_coords:
                dst_rank = process_mesh2.logical_coords_to_physical_ranks(
                    [dst_coord]
                )[0]
                self._inter_mesh_process_groups_dp[(src_rank, dst_rank)] = True

        # ep-related process groups
        # mesh1:tp-ep-dp
        etp1 = process_mesh1.get_parallel_size("tp", independent_ep=True)
        ep1 = process_mesh1.get_parallel_size("ep", independent_ep=True)
        edp1 = process_mesh1.get_parallel_size("dp", independent_ep=True)
        # mesh2:dp
        edp2 = process_mesh2.get_parallel_size("dp", independent_ep=True)
        edp_overlapped_mapping = find_overlapped_mapping(edp1, edp2)
        src_pp_dims = [process_mesh1.get_parallel_size("pp") - 1]
        dst_pp_dims = [0]

        # find tp+ep+pp group connection
        for k in range(edp1):
            src_coord = [etp1 - 1, cp1-1, ep1 - 1, k, src_pp_dims[0]]
            dst_edp_dims = [dim for dim, _, _ in edp_overlapped_mapping[k]]
            dst_coords = list(
                itertools.product([0], [0], [0], dst_edp_dims, dst_pp_dims)
            )
            src_rank = process_mesh1.logical_coords_to_physical_ranks(
                [src_coord], independent_ep=True
            )[0]
            for dst_coord in dst_coords:
                dst_rank = process_mesh2.logical_coords_to_physical_ranks(
                    [dst_coord], independent_ep=True
                )[0]
                self._inter_mesh_process_groups_edp[(src_rank, dst_rank)] = True

    def build_all_inter_mesh_process_groups(self):
        if len(self._process_meshes) == 1:
            return

        for i in range(len(self._process_meshes) - 1):
            self.build_inter_mesh_process_groups_tp_cp(
                self._process_meshes[i], self._process_meshes[i + 1]
            )

    def build_global_process_groups(self):
        # build global model parallel process groups
        for mesh_index, process_mesh in enumerate(self._process_meshes):
            ranks_list = process_mesh.get_all_process_group_ranks(
                "tp-pp", independent_ep=False, check_initialized=True
            )
            ranks = list(itertools.chain.from_iterable(ranks_list))
            self._all_group_ranks["mp"].append(ranks)
            group = torch.distributed.new_group(ranks, timeout=self._timeout)
            if self._rank in ranks:
                self._group_ranks["mp"] = ranks
                self._process_groups["mp"].append(group)
                self._process_group_to_ranks[group] = ranks

            ranks_list = process_mesh.get_all_process_group_ranks(
                "tp-ep-pp", independent_ep=True, check_initialized=True
            )
            ranks = list(itertools.chain.from_iterable(ranks_list))
            self._all_group_ranks["mp_exp"].append(ranks)
            group = torch.distributed.new_group(ranks, timeout=self._timeout)
            if self._rank in ranks:
                self._group_ranks["mp_exp"] = ranks
                self._process_groups["mp_exp"].append(group)
                self._process_group_to_ranks[group] = ranks
            ranks_list = process_mesh.get_all_process_group_ranks(
                "pp", independent_ep=False, check_initialized=True
            )
            ranks = list(itertools.chain.from_iterable(ranks_list))
            if "last_rank" not in self._parallel_ranks:
                self._parallel_ranks["last_rank"] = []
            self._parallel_ranks["last_rank"].append(ranks[-1])
        # build global pipeline process groups
        def _backtrack(mesh_index, prev_rank, path, token = "pp", independent_ep=False):
            assert token in ["tp-pp", "pp", "tp-ep-pp"], f"Invalid token: {token} for inter-mesh process groups"
            group_name = get_group_name(token, independent_ep=independent_ep)
            if mesh_index == len(self._process_meshes):
                aggregated_ranks = [rank for ranks in path for rank in ranks]
                self._all_group_ranks[group_name].append(aggregated_ranks)
                group = torch.distributed.new_group(aggregated_ranks, timeout=self._timeout)
                if self._rank in aggregated_ranks:
                    self._process_groups[group_name].append(group)
                    self._group_ranks[group_name].append(aggregated_ranks)
                    self._process_group_to_ranks[group] = aggregated_ranks
                return
            current_mesh = self._process_meshes[mesh_index]
            ranks_list = current_mesh.get_all_process_group_ranks(token, independent_ep=independent_ep, check_initialized=True)
            valid_ranks_list = []
            for ranks in ranks_list:
                mesh_is_connected = False
                for prev_path_ranks in path:
                    for prev_path_rank in prev_path_ranks:
                        if token == "pp" and (prev_path_rank, ranks[0]) in self._inter_mesh_process_groups_pp:
                            mesh_is_connected = True
                        elif token == "tp-pp" and (prev_path_rank, ranks[0]) in self._inter_mesh_process_groups_dp:
                            mesh_is_connected = True
                        elif token == "tp-ep-pp" and (prev_path_rank, ranks[0]) in self._inter_mesh_process_groups_edp:
                            mesh_is_connected = True
                if prev_rank == -1 or mesh_is_connected:
                    valid_ranks_list.append(ranks)
            for ranks in valid_ranks_list:
                path.append(ranks)
                _backtrack(mesh_index + 1, ranks[-1], path, token=token, independent_ep=independent_ep)
                path.pop()
        # build the global process groups which across the different Processmesh
        _backtrack(0, -1, path=[], token="tp-pp", independent_ep=False)
        _backtrack(0, -1, path=[], token="pp", independent_ep=False)
        _backtrack(0, -1, path=[], token="tp-ep-pp", independent_ep=True)

        # 'last_rank' is the last rank of the last pipeline stage
        pp_ranks = self.get_global_all_group_ranks("pp")
        self._parallel_ranks["last_rank"] = pp_ranks[-1][-1] if isinstance(pp_ranks[0], list) else pp_ranks[-1]
        # build global embedding process groups
        for ranks in self._group_ranks["pp"]:
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                # `pp_split` is similar to the `pipeline_model_parallel_split_rank` in parallel_state
                if "pp_split" in self._parallel_ranks.keys() and self._parallel_ranks["pp_split"] is not None:
                    split_rank = self._parallel_ranks["pp_split"]
                    if ranks[split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[split_rank],
                            ranks[-1],
                        ]
                    if ranks[split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [
                            ranks[0],
                            ranks[split_rank],
                        ]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks
            group = torch.distributed.new_group(
                embedding_ranks, timeout=self._timeout
            )
            if self._rank in embedding_ranks and embedding_ranks and ("embd" not in self._group_ranks or embedding_ranks not in self._group_ranks["embd"]):
                self._process_groups["embd"].append(group)
                self._process_group_to_ranks[group] = embedding_ranks
                self._group_ranks["embd"].append(embedding_ranks)
            
            # if self._rank in ranks:
            #     self._group_ranks["embd"].append(embedding_ranks)

            group = torch.distributed.new_group(
                position_embedding_ranks, timeout=self._timeout
            )
            if self._rank in position_embedding_ranks:
                self._process_groups["embd_pos"].append(group)
                self._process_group_to_ranks[group] = position_embedding_ranks

            if self._rank in ranks:
                self._group_ranks["embd_pos"].append(position_embedding_ranks)

    def get_inter_mesh_process_group(self, src_rank, dst_rank):
        if (src_rank, dst_rank) in self._inter_mesh_process_groups_pp:
            return self._inter_mesh_process_groups_pp[(src_rank, dst_rank)]
        elif (dst_rank, src_rank) in self._inter_mesh_process_groups_pp:
            return self._inter_mesh_process_groups_pp[(dst_rank, src_rank)]
        else:
            raise RuntimeError(
                f"ProcessGroup [{src_rank}, {dst_rank}] does not exist."
            )
        
    def get_inter_mesh_tensor_slices_tp_cp(self, rank, local_tensor_shape, next=True):
        if (rank, local_tensor_shape, next) in self._inter_mesh_tensor_slices:
            return self._inter_mesh_tensor_slices[(rank, local_tensor_shape, next)]
        process_mesh1 = self._process_meshes[self._current_process_mesh_index]
        if next:
            process_mesh2 = self.get_next_process_mesh()
            # first stage of the next process mesh
            src_pp_dims = [process_mesh1.get_parallel_size("pp", independent_ep=False) - 1]
            dst_pp_dims = [0]
        else:
            process_mesh2 = self.get_prev_process_mesh()
            # last stage of the previous process mesh
            src_pp_dims = [0]
            dst_pp_dims = [process_mesh2.get_parallel_size("pp") - 1]
        tp1 = process_mesh1.get_parallel_size("tp", independent_ep=False)
        cp1 = process_mesh1.get_parallel_size("cp", independent_ep=False)
        dp1 = process_mesh1.get_parallel_size("dp", independent_ep=False)
        tp2 = process_mesh2.get_parallel_size("tp", independent_ep=False)
        cp2 = process_mesh2.get_parallel_size("cp", independent_ep=False)
        dp2 = process_mesh2.get_parallel_size("dp", independent_ep=False)
        # For now, we only support the case where the sequence dim is different.
        # However, the following code can be easily extended to support other cases.
        # assert dp1 == dp2, "Data parallel size should be the same."
        # Assume that the tensor shape is (seq_len, batch_size, hidden_size)
        local_seq_len, local_batch_size, local_hidden_size = local_tensor_shape
        if self._args.sequence_parallel:
            global_seq_len = local_seq_len * tp1 * cp1
        else:
            global_seq_len = local_seq_len * cp1
        global_batch_size = local_batch_size * dp1

        if self._args.sequence_parallel:
            chunks = math.lcm(tp1*cp1*2, tp2*cp2*2)
        else:
            chunks = math.lcm(cp1*2, cp2*2)
             # key: tp_rank + cp_rank * tp_size, value: ﹞???那y?Y
        sequence_mesh1 = self.distribute_data(tp1, cp1, chunks, sequence_parallel=self._args.sequence_parallel)
        sequence_mesh2 = self.distribute_data(tp2, cp2, chunks, sequence_parallel=self._args.sequence_parallel)
        if self._args.sequence_parallel:
            sp_overlapped_mapping = find_overlapped_mapping_tp_cp(sequence_mesh1, sequence_mesh2, global_seq_len, sequence_parallel=True)
        else:
            sp_overlapped_mapping = find_overlapped_mapping_tp_cp(sequence_mesh1, sequence_mesh2, global_seq_len, sequence_parallel=False, cp_size1=cp1, tp_size1=tp1, tp_size2=tp2)
        dp_overlapped_mapping = find_overlapped_mapping(dp1, dp2, global_batch_size)
        # find sp overlapped mapping


        for src_sp_dim in sp_overlapped_mapping.keys():
            src_tp_rank, src_cp_rank = src_sp_dim % tp1, src_sp_dim // tp1
            for src_dp_dim in range(dp1):
                src_coord = [src_tp_rank, src_cp_rank, src_dp_dim, src_pp_dims[0]]

                # --------11?“米㊣?～src_sp_dim && src_dp_dim??車|米??迄車D℅?㊣那車3谷?1??米--------
                # 米㊣?～src_sp_dim ??車|米??迄車Ddst_sp_dim
                dst_sp_dims = [c for c, _    in sp_overlapped_mapping[src_sp_dim]]
                # 米㊣?～src_dp_dim ??車|米??迄車Ddst_dp_dim
                dst_dp_dims = [c for c, _, _ in dp_overlapped_mapping[src_dp_dim]]
                dst_coords = list(
                    itertools.product(dst_sp_dims, dst_dp_dims, dst_pp_dims)
                )

                # ?????a??㏒o[(src_sp_start1, src_sp_end1), (src_sp_start2, src_sp_end2㏒?,...]
                # ?迆src_rank--->dst_rank赤“D?那㊣, src_rank﹞⊿?赤米?sequnce那y?Y2?芍?D?, 辰辰∩??芍車D?角????
                src_sp_start_and_end = [sp_overlap_start_end_lst for _, sp_overlap_start_end_lst in sp_overlapped_mapping[src_sp_dim]]
                # ?????a??: (src_dp_start, src_dp_end)
                drc_dp_start_and_end = [(s, e) for _, s, e in dp_overlapped_mapping[src_dp_dim]]

                sp_dp_pairs = list(itertools.product(src_sp_start_and_end, drc_dp_start_and_end))
                
                src_rank = process_mesh1.logical_coords_to_physical_ranks([src_coord])[0]
                for i, dst_coord in enumerate(dst_coords):
                    dst_sp_dim, dst_dp_dim, dst_pp_dim = dst_coord
                    dst_coord = [dst_sp_dim % tp2, dst_sp_dim // tp2, dst_dp_dim, dst_pp_dim]
                    dst_rank = process_mesh2.logical_coords_to_physical_ranks([dst_coord])[0]

                    sp_overlap_start_end_lst, (dp_start, dp_end) = sp_dp_pairs[i]

                    if (src_rank, local_tensor_shape, next) not in self._inter_mesh_tensor_slices: 
                        self._inter_mesh_tensor_slices[(src_rank, local_tensor_shape, next)] = []
                    self._inter_mesh_tensor_slices[
                        (src_rank, local_tensor_shape, next)
                    ].append(
                        (
                            dst_rank,
                            (dp_start, dp_end),
                            sp_overlap_start_end_lst,
                            local_hidden_size
                        )
                    )
        return self._inter_mesh_tensor_slices[(rank, local_tensor_shape, next)]
    

    def get_inter_mesh_tensor_slices(self, rank, local_tensor_shape, next=True):
        if (rank, local_tensor_shape, next) in self._inter_mesh_tensor_slices:
            return self._inter_mesh_tensor_slices[(rank, local_tensor_shape, next)]
        process_mesh1 = self._process_meshes[self._current_process_mesh_index]
        if next:
            process_mesh2 = self.get_next_process_mesh()
            # first stage of the next process mesh
            src_pp_dims = [process_mesh1.get_parallel_size("pp") - 1]
            dst_pp_dims = [0]
        else:
            process_mesh2 = self.get_prev_process_mesh()
            # last stage of the previous process mesh
            src_pp_dims = [0]
            dst_pp_dims = [process_mesh2.get_parallel_size("pp") - 1]
        tp1 = process_mesh1.get_parallel_size("tp", independent_ep=False)
        cp1 = process_mesh1.get_parallel_size("cp", independent_ep=False)
        dp1 = process_mesh1.get_parallel_size("dp", independent_ep=False)
        tp2 = process_mesh2.get_parallel_size("tp", independent_ep=False)
        cp2 = process_mesh2.get_parallel_size("cp", independent_ep=False)
        dp2 = process_mesh2.get_parallel_size("dp", independent_ep=False)
        # For now, we only support the case where the sequence dim is different.
        # However, the following code can be easily extended to support other cases.
        assert dp1 == dp2, "Data parallel size should be the same."
        # Assume that the tensor shape is (seq_len, batch_size, hidden_size)
        local_seq_len, local_batch_size, local_hidden_size = local_tensor_shape
        if not(tp1 == 1 and tp2 == 1):
            global_seq_len = local_seq_len * tp1 * cp1
            sp1 = tp1 * cp1
            sp2 = tp2 * cp2
        else:
            global_seq_len = local_seq_len * cp1 
            sp1 = cp1
            sp2 = cp2
        global_batch_size = local_batch_size * dp1
        sp_overlapped_mapping = find_overlapped_mapping(sp1, sp2, global_seq_len)
        dp_overlapped_mapping = find_overlapped_mapping(dp1, dp2, global_batch_size)
        tp_overlapped_mapping = None
        if not self._args.sequence_parallel:
            tp_overlapped_mapping = find_overlapped_mapping(tp1, tp2)
            
        for s in range(sp1):
            src_i, src_j = s % tp1, s // tp1
            for k in range(dp1):
                src_coord = [src_i, src_j, k, src_pp_dims[0]]
                
                valid_dst_sp_dims = []
                for dim, _, _ in sp_overlapped_mapping[s]:
                    if not self._args.sequence_parallel and tp_overlapped_mapping is not None:
                        src_tp_i = s % tp1
                        dst_tp_i = dim % tp2
                        valid_dst_tps = [t for t, _, _ in tp_overlapped_mapping[src_tp_i]]
                        if dst_tp_i not in valid_dst_tps:
                            continue
                    valid_dst_sp_dims.append(dim)

                dst_dp_dims = [c for c, _, _ in dp_overlapped_mapping[k]]
                dst_coords = list(
                    itertools.product(valid_dst_sp_dims, dst_dp_dims, dst_pp_dims)
                )
                src_sp_starts = [s for _, s, _ in sp_overlapped_mapping[s]]
                src_dp_starts = [s for _, s, _ in dp_overlapped_mapping[k]]
                src_starts = list(itertools.product(src_sp_starts, src_dp_starts))
                src_sp_ends = [e for _, _, e in sp_overlapped_mapping[s]]
                src_dp_ends = [e for _, _, e in dp_overlapped_mapping[k]]
                src_ends = list(itertools.product(src_sp_ends, src_dp_ends))
                src_rank = process_mesh1.logical_coords_to_physical_ranks([src_coord])[0]
                for i, dst_coord in enumerate(dst_coords):
                    sp_dim, dp_dim, pp_dim = dst_coord
                    dst_coord = [sp_dim % tp2, sp_dim // tp2, dp_dim, pp_dim]
                    dst_rank = process_mesh2.logical_coords_to_physical_ranks([dst_coord])[0]
                    sp_start, dp_start = src_starts[i]
                    sp_end, dp_end = src_ends[i]
                    if (src_rank, local_tensor_shape, next) not in self._inter_mesh_tensor_slices: 
                        self._inter_mesh_tensor_slices[(src_rank, local_tensor_shape, next)] = []
                    self._inter_mesh_tensor_slices[
                        (src_rank, local_tensor_shape, next)
                    ].append(
                        (
                            dst_rank,
                            (dp_start, dp_end),
                            (sp_start, sp_end),
                            local_hidden_size,
                        )
                    )
        return self._inter_mesh_tensor_slices[(rank, local_tensor_shape, next)]

    def get_current_process_mesh(self):
        assert self._current_process_mesh_index < len(self._process_meshes)
        return self._process_meshes[self._current_process_mesh_index]

    def get_prev_process_mesh(self):
        assert self._current_process_mesh_index - 1 >= 0 
        return self._process_meshes[self._current_process_mesh_index - 1]

    def get_next_process_mesh(self):
        assert self._current_process_mesh_index + 1 < len(self._process_meshes)
        return self._process_meshes[self._current_process_mesh_index + 1]

    def get_global_all_group_ranks(self, token, independent_ep=False, check_initialized=False):
        group_name = get_group_name(token, independent_ep=independent_ep)
        ranks = self._all_group_ranks.get(group_name, None)
        if check_initialized:
            assert (
                ranks is not None
            ), f"Process group {group_name} is not initialized."
        return ranks

    def get_model_parallel_group(self, with_expert_parallel=False):
        """Get the model parallel group the caller rank belongs to.
        Returns the list of process groups (one per mesh the current rank belongs to).
        """
        if with_expert_parallel:
            group = self._process_groups.get("mp_exp", None)
            assert group is not None and len(group) > 0, "model parallel group is not initialized"
            return group
        group = self._process_groups.get("mp", None)
        assert group is not None and len(group) > 0, 'model parallel group is not initialized'
        return group

    def get_tensor_model_parallel_group(self, check_initialized=True):
        """Get the tensor model parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "tp", independent_ep=False, gloo=False, check_initialized=check_initialized
        )

    def get_pipeline_model_parallel_group(self, check_initialized=True):
        """Get the pipeline model parallel group the caller rank belongs to.
        Returns the list of process groups (one per pipeline the current rank belongs to).
        """
        group = self._process_groups.get("pp", None)
        assert group is not None and len(group) > 0, "pipeline_model parallel group is not initialized"
        return group

    def get_data_parallel_group(self, with_context_parallel=False):
        """Get the data parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            return current_process_mesh.get_process_group(
                "dp-cp", independent_ep=False, gloo=False, check_initialized=True
            )
        else:
            return current_process_mesh.get_process_group(
                "dp", independent_ep=False, gloo=False, check_initialized=True
            )

    def get_data_parallel_group_gloo(self, with_context_parallel=False):
        """Get the data parallel group-gloo the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            return current_process_mesh.get_process_group(
                "dp-cp", independent_ep=False, gloo=True, check_initialized=True
            )
        else:
            return current_process_mesh.get_process_group(
                "dp", independent_ep=False, gloo=True, check_initialized=True
            )

    def get_context_parallel_group(self, check_initialized=True):
        """Get the context parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "cp", independent_ep=False, gloo=False, check_initialized=check_initialized
        )

    def get_context_parallel_global_ranks(self, check_initialized=True):
        """Get all global ranks of the context parallel group that the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group_ranks(
            "cp", independent_ep=False, check_initialized=check_initialized
        )
    
    # def get_ulysses_sp_parallel_group(self, check_initialized=True):
    #     """Get the ulysses sequence parallel group the caller rank belongs to."""
    #     current_process_mesh = self._process_meshes[self._current_process_mesh_index]
    #     return current_process_mesh.get_process_group(
    #         "usp", independent_ep=False, gloo=False, check_initialized=check_initialized
    #     )
        
    # def get_ulysses_sp_parallel_global_ranks(self, check_initialized=True):
    #     """Get all global ranks of the ulysses sequence parallel group that the caller rank belongs to."""
    #     current_process_mesh = self._process_meshes[self._current_process_mesh_index]
    #     return current_process_mesh.get_process_group_ranks(
    #         "usp", independent_ep=False, check_initialized=check_initialized
    #     )

    def get_embedding_group(self):
        """Get the embedding group the caller rank belongs to."""
        groups = self._process_groups.get("embd", None)
        assert groups is not None, 'embedding group is not initialized'
        for group in groups:
            if self._rank in self._process_group_to_ranks[group]:
                embd_group = group
                break
        return embd_group 

    def get_position_embedding_group(self):
        """Get the position embedding group the caller rank belongs to."""
        groups = self._process_groups.get("embd_pos", None)
        assert groups is not None, 'Position embedding group is not initialized'
        for group in groups:
            if self._rank in self._process_group_to_ranks[group]:
                pos_embd_group = group
                break
        return pos_embd_group

    def get_amax_reduction_group(self, with_context_parallel=False):
        """Get the FP8 amax reduction group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            return current_process_mesh.get_process_group(
                "tp-dp-cp", independent_ep=False, gloo=False, check_initialized=True
            )
        else:
            return current_process_mesh.get_process_group(
                "tp-dp", independent_ep=False, gloo=False, check_initialized=True
            )

    def get_tensor_and_data_parallel_group(self, with_context_parallel=False):
        """Get the tensor and data parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            return current_process_mesh.get_process_group(
                "tp-dp-cp", independent_ep=False, gloo=False, check_initialized=True
            )
        else:
            return current_process_mesh.get_process_group(
                "tp-dp", independent_ep=False, gloo=False, check_initialized=True
            )

    def get_tensor_and_context_parallel_group(self):
        """Get the tensor- and context-parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "tp-cp", independent_ep=False, gloo=False, check_initialized=True
        )

    def get_expert_model_parallel_group(self):
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "ep", independent_ep=True, gloo=False, check_initialized=True
        )

    def get_tensor_and_expert_parallel_group(self):
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "tp-ep", independent_ep=True, gloo=False, check_initialized=True
        )

    def get_data_modulo_expert_parallel_group(self, with_context_parallel=False):
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            return current_process_mesh.get_process_group(
                "dp-cp", independent_ep=True, gloo=False, check_initialized=True
            )
        else:
            
            return current_process_mesh.get_process_group(
                "dp", independent_ep=True, gloo=False, check_initialized=True
            )

    def get_data_modulo_expert_parallel_group_gloo(self):
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "dp", independent_ep=True, gloo=True, check_initialized=True
        )

    def set_expert_model_parallel_world_size(self, world_size):
        self._parallel_world_sizes["ep"] = world_size

    def set_tensor_model_parallel_world_size(self, world_size):
        """Set the tensor model parallel size"""
        self._parallel_world_sizes["tp"] = world_size

    def set_pipeline_model_parallel_world_size(self, world_size):
        """Set the pipeline model parallel size"""
        self._parallel_world_sizes["pp"] = world_size

    def set_virtual_pipeline_model_parallel_world_size(self, world_size):
        """Set the pipeline model parallel size"""
        self._parallel_world_sizes["vpp"] = world_size

    def get_tensor_model_parallel_world_size(self):
        """Return world size for the tensor model parallel group."""
        size = self._parallel_world_sizes.get("tp", None)
        if size is not None:
            return size
        return torch.distributed.get_world_size(group=self.get_tensor_model_parallel_group())

    def get_pipeline_model_parallel_world_size(self, group=None):
        """Return world size for the pipeline model parallel group."""
        size = self._parallel_world_sizes.get("pp", None)
        if size is not None:
            return size
        if group is None:
            group = self.get_pipeline_model_parallel_group()
        # get_pipeline_model_parallel_group() returns a list; take the first element
        if isinstance(group, list):
            group = group[0]
        return torch.distributed.get_world_size(group)

    def set_expert_model_parallel_rank(self, rank):
        """Set expert model parallel rank."""
        self._parallel_ranks["ep"] = rank

    def set_tensor_model_parallel_rank(self, rank):
        """Set tensor model parallel rank."""
        self._parallel_ranks["tp"] = rank

    def set_pipeline_model_parallel_rank(self, rank):
        """Set pipeline model parallel rank."""
        self._parallel_ranks["pp"] = rank

    def set_pipeline_model_parallel_split_rank(self, rank):
        """Set pipeline model parallel split rank."""
        self._parallel_ranks["pp-split"] = rank

    def get_tensor_model_parallel_rank(self):
        """Return my rank for the tensor model parallel group."""
        rank = self._parallel_ranks.get("tp", None)
        if rank is not None:
            return rank
        return torch.distributed.get_rank(group=self.get_tensor_model_parallel_group())

    def get_pipeline_model_parallel_rank(self, group=None):
        """Return my rank for the pipeline model parallel group."""
        rank = self._parallel_ranks.get("pp", None)
        if rank is not None:
            return rank
        if group is None:
            group = self.get_pipeline_model_parallel_group()
        # get_pipeline_model_parallel_group() returns a list; take the first element
        if isinstance(group, list):
            group = group[0]
        return torch.distributed.get_rank(group=group)

    def get_pipeline_model_parallel_split_rank(self):
        """Return pipeline model parallel split rank."""
        return self._parallel_ranks.get("pp-split", None)

    def is_pipeline_first_stage(self, ignore_virtual=False, group=None):
        """Return True if in the first pipeline model-parallel stage, False otherwise."""
        if not ignore_virtual:
            if (
                self.get_virtual_pipeline_model_parallel_world_size() is not None
                and self.get_virtual_pipeline_model_parallel_rank() != 0
            ):
                return False
        return self.get_pipeline_model_parallel_rank(group) == 0

    def is_pipeline_last_stage(self, ignore_virtual=False, group=None):
        """Return True if in the last pipeline model-parallel stage, False otherwise."""
        if not ignore_virtual:
            virtual_pipeline_model_parallel_world_size = (
                self.get_virtual_pipeline_model_parallel_world_size()
            )
            if (
                virtual_pipeline_model_parallel_world_size is not None
                and self.get_virtual_pipeline_model_parallel_rank()
                != (virtual_pipeline_model_parallel_world_size - 1)
            ):
                return False
        return self.get_pipeline_model_parallel_rank(group) == (self.get_pipeline_model_parallel_world_size(group) - 1)

    def is_rank_in_embedding_group(self, ignore_virtual=False, group=None):
        """Return true if current rank is in embedding group, False otherwise."""
        rank = torch.distributed.get_rank()
        if group is None:
            group = self._process_groups.get("embd", None)
            if group is None:
                return False
            else:
                group = group[0]
        ranks = self._process_group_to_ranks[group]
        if ignore_virtual:
            return rank in ranks 
        if rank in ranks:
            if rank == ranks[0]:
                return self.is_pipeline_first_stage(ignore_virtual=False, group=group)
            elif rank == ranks[-1]:
                return self.is_pipeline_last_stage(ignore_virtual=False, group=group)
            else:
                return True
        return False

    def is_rank_in_position_embedding_group(self, group=None):
        """Return true if current rank is in position embedding group, False otherwise."""
        rank = torch.distributed.get_rank()
        if group is None:
            group = self._process_groups.get("embd_pos", None)
            if group is None:
                return False
            else:
                group = group[0]
        ranks = self._process_group_to_ranks[group]
        return rank in ranks 

    def is_pipeline_stage_before_split(self, rank=None, group=None):
        """Return True if pipeline stage executes encoder block for a model
        with both encoder and decoder."""
        if self.get_pipeline_model_parallel_world_size(group) == 1:
            return True
        if rank is None:
            rank = self.get_pipeline_model_parallel_rank(group)
        split_rank = self.get_pipeline_model_parallel_split_rank()
        if split_rank is None:
            return True
        if rank < split_rank:
            return True
        return False

    def is_pipeline_stage_after_split(self, rank=None, group=None):
        """Return True if pipeline stage executes decoder block for a model
        with both encoder and decoder."""
        if self.get_pipeline_model_parallel_world_size(group) == 1:
            return True
        if rank is None:
            rank = self.get_pipeline_model_parallel_rank(group)
        split_rank = self.get_pipeline_model_parallel_split_rank()
        if split_rank is None:
            return True
        if rank >= split_rank:
            return True
        return False

    def is_pipeline_stage_at_split(self, group=None):
        """Return true if pipeline stage executes decoder block and next
        stage executes encoder block for a model with both encoder and
        decoder."""
        rank = self.get_pipeline_model_parallel_rank(group)
        return self.is_pipeline_stage_before_split(rank, group) and self.is_pipeline_stage_after_split(rank + 1, group)

    def get_virtual_pipeline_model_parallel_rank(self):
        """Return the virtual pipeline-parallel rank."""
        return self._parallel_ranks.get("vpp", None)

    def set_virtual_pipeline_model_parallel_rank(self, rank):
        """Set the virtual pipeline-parallel rank."""
        self._parallel_ranks["vpp"] = rank

    def get_virtual_pipeline_model_parallel_world_size(self):
        """Return the virtual pipeline-parallel world size."""
        return self._parallel_world_sizes.get("vpp", None)

    def get_tensor_model_parallel_src_rank(self):
        """Calculate the global rank corresponding to the first local rank
        in the tensor model parallel group."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        ranks = current_process_mesh.get_process_group_ranks(
            "tp", independent_ep=False, check_initialized=True
        )
        return ranks[0]

    def get_data_parallel_src_rank(self, with_context_parallel=False):
        """Calculate the global rank corresponding to the first local rank
        in the data parallel group."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            ranks = current_process_mesh.get_process_group_ranks(
                "dp-cp", independent_ep=False, check_initialized=True
            )
        else:
            ranks = current_process_mesh.get_process_group_ranks(
                "dp", independent_ep=False, check_initialized=True
            )
        return ranks[0]

    def get_pipeline_model_parallel_first_rank(self, group=None):
        """Return the global rank of the first process in the pipeline for the
        current tensor parallel group"""
        if group is None:
            group = self.get_pipeline_model_parallel_group()
        if isinstance(group, list):
            group = group[0]
        ranks = self._process_group_to_ranks.get(group, None)
        assert ranks is not None, "Pipeline parallel group is not initialized"
        return ranks[0]

    def get_pipeline_model_parallel_last_rank(self, group=None):
        """Return the global rank of the last process in the pipeline for the
        current tensor parallel group"""
        if group is None:
            group = self.get_pipeline_model_parallel_group()
        if isinstance(group, list):
            group = group[0]
        ranks = self._process_group_to_ranks.get(group, None)
        assert ranks is not None, "Pipeline parallel group is not initialized"
        last_rank_local = self.get_pipeline_model_parallel_world_size(group) - 1
        return ranks[last_rank_local]

    def get_pipeline_model_parallel_next_rank(self, group=None):
        """Return the global rank that follows the caller in the pipeline"""
        if group is None:
            group = self.get_pipeline_model_parallel_group()
        if isinstance(group, list):
            group = group[0]
        ranks = self._process_group_to_ranks.get(group, None)
        assert ranks is not None, "Pipeline parallel group is not initialized"
        rank_in_pipeline = self.get_pipeline_model_parallel_rank(group)
        world_size = self.get_pipeline_model_parallel_world_size(group)
        return ranks[(rank_in_pipeline + 1) % world_size]

    def get_pipeline_model_parallel_prev_rank(self, group=None):
        """Return the global rank that preceeds the caller in the pipeline"""
        if group is None:
            group = self.get_pipeline_model_parallel_group()
        if isinstance(group, list):
            group = group[0]
        ranks = self._process_group_to_ranks.get(group, None)
        assert ranks is not None, "Pipeline parallel group is not initialized"
        rank_in_pipeline = self.get_pipeline_model_parallel_rank(group)
        world_size = self.get_pipeline_model_parallel_world_size(group)
        return ranks[(rank_in_pipeline - 1) % world_size]

    def get_last_rank_when_using_pipeline(self):
        """Return the global rank of the last process in the pipeline"""
        assert (
            self._parallel_ranks.get("last_rank", None) is not None
        ), "Last rank when using pipeline is not initialized"
        return self._parallel_ranks["last_rank"][self._current_process_mesh_index]

    def get_data_parallel_world_size(self, with_context_parallel=False):
        """Return world size for the data parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(
                group=self.get_data_parallel_group(with_context_parallel=with_context_parallel)
            )
        else:
            return 0

    def get_data_parallel_rank(self, with_context_parallel=False):
        """Return my rank for the data parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(
                group=self.get_data_parallel_group(with_context_parallel=with_context_parallel)
            )
        else:
            return 0

    def get_context_parallel_world_size(self):
        """Return world size for the context parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(group=self.get_context_parallel_group())
        else:
            return 0

    def get_context_parallel_rank(self):
        """Return my rank for the context parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_context_parallel_group())
        else:
            return 0
    
    # def get_ulysses_sp_parallel_world_size(self):
    #     """Return world size for the ulysses sequence parallel group."""
    #     if torch.distributed.is_available() and torch.distributed.is_initialized():
    #         return torch.distributed.get_world_size(group=self.get_ulysses_sp_parallel_group())
    #     else:
    #         return 0

    # def get_ulysses_sp_parallel_rank(self):
    #     """Return my rank for the ulysses sequence parallel group."""
    #     if torch.distributed.is_available() and torch.distributed.is_initialized():
    #         return torch.distributed.get_rank(group=self.get_ulysses_sp_parallel_group())
    #     else:
    #         return 0

    def get_expert_model_parallel_world_size(self):
        """Return world size for the expert model parallel group"""
        if self._parallel_world_sizes.get("ep", None) is not None:
            return self._parallel_world_sizes["ep"]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
                group=self.get_tensor_and_expert_parallel_group()
            )
            return tensor_and_expert_parallel_world_size // self.get_tensor_model_parallel_world_size()
        else:
            return 0

    def get_tensor_and_expert_parallel_world_size(self):
        """Return world size for the expert model parallel group times model parallel group.
           Currently, each expert will also be distributed across TP group by default.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
                group=self.get_tensor_and_expert_parallel_group()
            )
            return tensor_and_expert_parallel_world_size
        else:
            return 0

    def get_expert_model_parallel_rank(self):
        """Return my rank for the expert parallel group"""
        if self._parallel_ranks.get("ep", None) is not None:
            return self._parallel_ranks["ep"]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tensor_and_expert_parallel_rank = torch.distributed.get_rank(
                group=self.get_tensor_and_expert_parallel_group()
            )
            return tensor_and_expert_parallel_rank // self.get_tensor_model_parallel_world_size()
        else:
            return 0

    def get_data_modulo_expert_parallel_rank(self, with_context_parallel):
        """Return my rank for the context parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_data_modulo_expert_parallel_group(with_context_parallel=with_context_parallel))
        else:
            return 0

    def get_tensor_and_expert_parallel_rank(self):
        """Return my rank for the tensor and expert parallel group"""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_tensor_and_expert_parallel_group())
        else:
            return 0

    def set_global_memory_buffer(self):
        """Initialize global buffer"""
        assert self._global_memory_buffer is None, 'global memory buffer is already initialized'
        self._global_memory_buffer = GlobalMemoryBuffer()

    def get_global_memory_buffer(self):
        """Return the global GlobalMemoryBuffer object"""
        assert self._global_memory_buffer is not None, 'global memory buffer is not initialized'
        return self._global_memory_buffer

    def destroy_global_memory_buffer(self):
        """Sets the global memory buffer to None"""
        self._global_memory_buffer = None