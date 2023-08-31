import os
from argparse import ArgumentParser
from os.path import abspath, dirname, join


CURRENT_DIR = dirname(abspath(__file__))


def build_gunicorn_cmd_str(
    gunicorn_config_file_path: str,
    proc_name: str,
    log_dir: str,
    port: int
):
    access_log_file = join(log_dir, f"{proc_name}_access.log")
    error_log_file = join(log_dir, f"{proc_name}_error.log")
    cmd_str = (
        f'-c "{gunicorn_config_file_path}" '
        f'--bind "0.0.0.0:{port}" '
        f'--name "{proc_name}" '
        f'--access-logfile "{access_log_file}" '
        f'--error-logfile "{error_log_file}" '
        f'-D'  # running in background
    )
    return cmd_str


def main():
    parser = ArgumentParser()

    # args to start client app
    parser.add_argument(
        "--start_client_app", action="store_true",
        help="whether to start client app"
    )
    parser.add_argument(
        "--client_config_file_path", type=str, default="client_config.json",
        help="local path to read client config file"
    )
    parser.add_argument(
        "--client_config_hot_update_interval_minutes", type=int, default=1,
        help="specify the interval minutes to hot update client config"
    )
    parser.add_argument(
        "--client_debug", action="store_true",
        help="whether to start client app in debug mode"
    )
    parser.add_argument(
        "--client_port", type=int, default=8000,
        help="the port the client app will use"
    )

    # args to start CB server app
    parser.add_argument(
        "--start_cb_server_app", action="store_true",
        help="whether to start CB server app"
    )
    parser.add_argument(
        "--cb_server_model_id", type=str,
        help="model id for the CB server that will be started"
    )
    parser.add_argument(
        "--cb_server_config_file_path", type=str, default="cb_server_config.json",
        help="local path to read CB server config file"
    )
    parser.add_argument(
        "--cb_server_debug", action="store_true",
        help="whether to start CB server app in debug mode"
    )
    parser.add_argument(
        "--cb_server_port", type=int, default=8001,
        help="the part the CB server app will use"
    )

    # args to start SB server app
    parser.add_argument(
        "--start_sb_server_app", action="store_true",
        help="whether to start SB server app"
    )
    parser.add_argument(
        "--sb_server_model_id", type=str,
        help="model id for the SB server that will be started"
    )
    parser.add_argument(
        "--sb_server_config_file_path", type=str, default="sb_server_config.json",
        help="local path to read SB server config file"
    )
    parser.add_argument(
        "--sb_server_debug", action="store_true",
        help="whether to start SB server app in debug mode"
    )
    parser.add_argument(
        "--sb_server_port", type=int, default=8002,
        help="the part the SB server app will use"
    )

    # args that are shared among apps
    parser.add_argument(
        "--client_url", type=str, default=None,
        help="URL of client, only be used by servers, has no effect for now"
    )
    parser.add_argument(
        "--gunicorn_config_file_path", type=str, default="gunicorn_config.py",
        help="local path to read a python script that stores some common gunicorn settings for APPs"
    )

    args = parser.parse_args()

    # start client app if triggered
    if args.start_client_app:
        cmd_str = (
            f"""gunicorn 'client_app:build_app("""
            f"""client_config_file_path="{args.client_config_file_path}","""
            f"""client_config_hot_update_interval_minutes="{args.client_config_hot_update_interval_minutes}","""
            f"""debug={args.client_debug})' """
        )
        cmd_str += build_gunicorn_cmd_str(
            gunicorn_config_file_path=args.gunicorn_config_file_path,
            proc_name="llm_inference_client",
            log_dir=join(CURRENT_DIR, "logs"),
            port=args.client_port
        )
        print(f"start client app, command being executed is:\n{cmd_str}")
        os.system(cmd_str)

    # start CB server app if triggered
    if args.start_cb_server_app:
        cmd_str = (
            f"""gunicorn 'continuous_batching_server_app:build_app("""
            f"""model_id="{args.cb_server_model_id}","""
            f"""server_config_file_path="{args.cb_server_config_file_path}","""
            # f"""client_url={args.client_url}"""
            f"""debug={args.cb_server_debug})' """
        )
        cmd_str += build_gunicorn_cmd_str(
            gunicorn_config_file_path=args.gunicorn_config_file_path,
            proc_name="llm_inference_cb_server",
            log_dir=join(CURRENT_DIR, "logs"),
            port=args.cb_server_port
        )
        print(f"start CB server app, command being executed is:\n{cmd_str}")
        os.system(cmd_str)

    # start SB server app if triggered
    if args.start_sb_server_app:
        cmd_str = (
            f"""gunicorn 'static_batching_server_app:build_app("""
            f"""model_id="{args.sb_server_model_id}","""
            f"""server_config_file_path="{args.sb_server_config_file_path}","""
            # f"""client_url={args.client_url}"""
            f"""debug={args.sb_server_debug})' """
        )
        cmd_str += build_gunicorn_cmd_str(
            gunicorn_config_file_path=args.gunicorn_config_file_path,
            proc_name="llm_inference_sb_server",
            log_dir=join(CURRENT_DIR, "logs"),
            port=args.sb_server_port
        )
        print(f"start SB server app, command being executed is:\n{cmd_str}")
        os.system(cmd_str)


if __name__ == "__main__":
    main()
