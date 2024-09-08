import subprocess


def run_terminal_command(command: str) -> None:
    """
    Execute a terminal command and print the output in real-time.

    :param str command: The command to be executed in the terminal

    :return: No returned data
    :rtype: None
    """

    with subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
    ) as process:
        while process.poll() is None:
            line = process.stdout.readline()
            print(line, end='')
