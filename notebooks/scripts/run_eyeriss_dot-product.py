profiler = Profiler(
            top_dir='workloads',
            sub_dir='nerf' if not is_sparse else 'nerf-sparse',
            timeloop_dir=f"designs/{arch}",
            arch_name=arch,
            model=model,
            input_size=(1, 3),
        )