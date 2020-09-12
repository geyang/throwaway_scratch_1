from rl import main, Args
from params_proto.neo_hyper import Sweep
from ml_logger import logger

if __name__ == '__main__':
    import jaynes
    from lp_analysis import instr

    env_names = [
        # Bin task set
        # 'fetch:Bin-pick-v0',
        # 'fetch:Bin-place-v0',
        # # Bin debug task sets
        # 'fetch:Bin-fixed-v0',
        # 'fetch:Bin-fixed-hide-v0',
        'fetch:Bin-fixed-pos-v0',
    ]

    with Sweep(Args) as sweep:

        Args.gamma = 0.99
        Args.clip_inputs = True
        Args.normalize_inputs = True
        Args.n_workers = 12
        Args.n_epochs = 500

        with sweep.product:
            with sweep.zip:
                Args.env_name = env_names

            Args.seed = [100, 200, 300, 400, 500, 600]

    # jaynes.config('local' if Args.debug else "gpu-vaughan", launch=dict(timeout=10000))
    jaynes.config('local' if Args.debug else "cpu-mars")
    # jaynes.config('local')
    for i, deps in sweep.items():
        thunk = instr(main, deps, _job_postfix=f"{Args.env_name.split(':')[-1]}")
        logger.log_text("""
            keys:
            - run.status
            - Args.env_name
            - host.hostname
            charts:
            - yKey: test/success
              xKey: epoch
            - yKey: test/success
              xKey: timesteps
            - yKey: train/success/mean
              xKey: time
            """, ".charts.yml", dedent=True)
        jaynes.run(thunk)
