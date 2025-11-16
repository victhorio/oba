def print_header(title: str) -> None:
    line_separator = "==" + "=" * len(title) + "=="
    line_title = f"= {title} ="

    print(_ANSI_YELLOW_BOLD)
    print(line_separator)
    print(line_title)
    print(line_separator)
    print(_ANSI_RESET)


def print_cost(value: float) -> None:
    print(_ANSI_YELLOW_BOLD)
    print(f"Cost: ${value:.3f}")
    print(_ANSI_RESET)


def print_result_header(title: str) -> None:
    print(_ANSI_GREEN_BOLD)
    print(f"-- {title} --")
    print(_ANSI_RESET)


def print_result_footer() -> None:
    print("\n")


_ANSI_RESET = "\033[0m"
_ANSI_YELLOW_BOLD = "\033[33;1m"
_ANSI_GREEN_BOLD = "\033[32;1m"
