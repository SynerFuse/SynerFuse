import warnings

from datetime import timedelta

import torch

try:
    import flagcx  # noqa: F401
except ImportError:
    warnings.warn(
        "flagcx is not installed, you can't use flagcx backend for communication.",
        ImportWarning,
    )

from synerfuse_hetero.train.hetero.parallel_context import RankMapper


class FSTrainArguments:
    """Extend the Megatron arguments with synerfuse_hetero specific arguments."""

    def __init__(self, args, rank_mapper=None):
        self.args = args
        self._rank_mapper = rank_mapper

    def __getattr__(self, name):
        if name == "rank_mapper":
            return self._rank_mapper
        return getattr(self.args, name)

    def _initialize_distributed(self):
        """Initialize torch.distributed before hetero argument validation."""
        args = self.args

        device_count = torch.cuda.device_count()
        if torch.distributed.is_initialized():
            if args.rank == 0:
                print(
                    "torch distributed is already initialized, skipping initialization ...",
                    flush=True,
                )
            args.rank = torch.distributed.get_rank()
            args.world_size = torch.distributed.get_world_size()
            return

        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)

        if device_count > 0:
            if args.local_rank is None:
                args.local_rank = args.rank % device_count
            torch.cuda.set_device(args.local_rank)

        init_process_group_kwargs = {
            "backend": args.distributed_backend,
            "world_size": args.world_size,
            "rank": args.rank,
            "timeout": timedelta(minutes=args.distributed_timeout_minutes),
        }
        if args.distributed_backend == "flagcx":
            init_process_group_kwargs["backend"] = "cpu:gloo,cuda:flagcx"
        if args.enable_hetero and args.hetero_use_cpu_communication:
            init_process_group_kwargs["backend"] = "cpu:gloo"
        torch.distributed.init_process_group(**init_process_group_kwargs)

    def _build_rank_mapper(self):
        self._initialize_distributed()
        self._rank_mapper = RankMapper(self.args)
        return self._rank_mapper

    def pre_validate_args(self):
        """Pre-validate hetero arguments before the regular argument validation."""
        if self._rank_mapper is None:
            self._build_rank_mapper()

        assert (
            self.args.hetero_process_meshes is not None
        ), "hetero_process_meshes should be specified when enable_hetero is True"
        assert (
            len(self.args.hetero_process_meshes) % 5 == 0
        ), (
            f"length of hetero_process_meshes {self.args.hetero_process_meshes} "
            "should be divisible by 5, the format should be "
            "tp0, cp0, ep0, dp0, pp0, tp1, cp1, ep1, dp1, pp1, ..."
        )

        hetero_process_meshes_tp = self.args.hetero_process_meshes[0::5]
        hetero_process_meshes_cp = self.args.hetero_process_meshes[1::5]
        hetero_process_meshes_ep = self.args.hetero_process_meshes[2::5]
        hetero_process_meshes_dp = self.args.hetero_process_meshes[3::5]
        hetero_process_meshes_pp = self.args.hetero_process_meshes[4::5]

        if self.expert_tensor_parallel_size_per_process_mesh is not None:
            assert len(self.expert_tensor_parallel_size_per_process_mesh) == len(
                hetero_process_meshes_tp
            ), (
                "length of expert_tensor_parallel_size_per_process_mesh "
                f"{len(self.expert_tensor_parallel_size_per_process_mesh)} should be "
                f"equal to length of hetero_process_meshes_tp {len(hetero_process_meshes_tp)}"
            )

        self.args.data_parallel_size = hetero_process_meshes_dp[0]
        assert all(
            self.args.data_parallel_size * self.args.micro_batch_size % hetero_dp == 0
            for hetero_dp in hetero_process_meshes_dp
        ), (
            f"data_parallel_size * micro_batch_size "
            f"{self.args.data_parallel_size * self.args.micro_batch_size} should be "
            f"divisible by all hetero_process_meshes_dp {hetero_process_meshes_dp}!"
        )

        for context_parallel_size in hetero_process_meshes_cp:
            if self.args.seq_length is not None and context_parallel_size > 1:
                assert self.args.seq_length % (context_parallel_size * 2) == 0, (
                    "seq-length should be a multiple of 2 * context-parallel-size "
                    "if context-parallel-size > 1."
                )
            if context_parallel_size > 1:
                assert not self.args.use_legacy_models, (
                    "Context parallelism is not supported in legacy models."
                )
            if self.args.use_tp_pp_dp_mapping:
                assert context_parallel_size * self.args.expert_model_parallel_size <= 1, (
                    "context_parallel and expert_model_parallel can't be used with "
                    "tp-pp-dp mapping."
                )

        assert all(1 == hetero_ep for hetero_ep in hetero_process_meshes_ep) or all(
            1 != hetero_ep for hetero_ep in hetero_process_meshes_ep
        ), (
            f"all hetero_process_meshes_ep {hetero_process_meshes_ep} should be 1 "
            "or none of hetero_process_meshes_ep should be 1!"
        )

        assert self.args.pipeline_model_parallel_size == sum(hetero_process_meshes_pp), (
            f"origin pipeline_model_parallel_size {self.args.pipeline_model_parallel_size} "
            f"should match sum of hetero_process_meshes_pp {hetero_process_meshes_pp}!"
        )
        assert (
            self.args.pipeline_model_parallel_split_rank is None
        ), "pipeline_model_parallel_split_rank not supported with process_meshes set!"
        self.args.transformer_pipeline_model_parallel_size = (
            self.args.pipeline_model_parallel_size
        )

        if self.args.untie_embeddings_and_output_weights is False or getattr(
            self.args, "mtp_num_layers", 0
        ):
            assert hetero_process_meshes_tp[0] == hetero_process_meshes_tp[-1], (
                "if untie_embeddings_and_output_weights is False or mtp_num_layers "
                "is not 0, the first and last stage should have the same tp degree!"
            )

            if (
                hetero_process_meshes_dp[0] != hetero_process_meshes_dp[-1]
                and self.args.use_distributed_optimizer
            ):
                assert (
                    hetero_process_meshes_dp[0] % hetero_process_meshes_dp[-1] == 0
                    or hetero_process_meshes_dp[-1] % hetero_process_meshes_dp[0] == 0
                ), (
                    "if untie_embeddings_and_output_weights is False and "
                    "hetero_process_meshes_dp[0] and hetero_process_meshes_dp[-1] "
                    "are different, one should be divisible by the other currently!"
                )
                assert getattr(self.args, "use_partial_reduce_for_shared_embedding", False) is True, (
                    "if untie_embeddings_and_output_weights is False and "
                    "hetero_process_meshes_dp[0] and hetero_process_meshes_dp[-1] "
                    "are different, use_partial_reduce_for_shared_embedding should be True!"
                )

        if self.args.enable_hetero:
            assert (
                self.args.num_layers_per_virtual_pipeline_stage is None
            ), "virtual pipeline not support now!"

        hetero_process_meshes = []
        for i in range(0, len(self.args.hetero_process_meshes), 5):
            hetero_process_meshes.append(self.args.hetero_process_meshes[i : i + 5])
        self.args.hetero_process_meshes = hetero_process_meshes

        assert len(hetero_process_meshes) == len(self.args.hetero_device_types), (
            f"length of hetero_process_meshes {len(hetero_process_meshes)} should "
            f"match length of hetero_device_types {len(self.args.hetero_device_types)}"
        )
        assert self.args.hetero_current_device_type in self.args.hetero_device_types, (
            f"hetero_current_device_type {self.args.hetero_current_device_type} "
            f"should be in hetero_device_types {self.args.hetero_device_types}"
        )

        accumulated_world_size = 0
        logical_rank = self.rank_mapper.to_logical_ranks([torch.distributed.get_rank()])[0]
        for tp, cp, _, dp, pp in self.args.hetero_process_meshes:
            temp_world_size = tp * cp * dp * pp
            if accumulated_world_size <= logical_rank < accumulated_world_size + temp_world_size:
                self.args.micro_batch_size = (
                    self.args.data_parallel_size * self.args.micro_batch_size // dp
                )
                break
            accumulated_world_size += temp_world_size
