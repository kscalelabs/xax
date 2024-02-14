"""Defines functions for parallelism."""

import os

_RANK: int | None = None
_LOCAL_RANK: int | None = None
_WORLD_SIZE: int | None = None
_LOCAL_WORLD_SIZE: int | None = None
_MASTER_ADDR: str | None = None
_MASTER_PORT: int | None = None
_INIT_METHOD: str | None = None


def set_rank(rank: int) -> None:
    global _RANK

    if rank != _RANK:
        _RANK = rank
        os.environ["RANK"] = str(rank)
    else:
        raise ValueError(f"Rank {rank} is already set")


def get_rank_optional() -> int | None:
    return _RANK


def get_rank() -> int:
    return 0 if _RANK is None else _RANK


def clear_rank() -> None:
    global _RANK

    _RANK = None
    os.environ.pop("RANK", None)


def set_local_rank(rank: int) -> None:
    global _LOCAL_RANK

    if rank != _LOCAL_RANK:
        _LOCAL_RANK = rank
        os.environ["LOCAL_RANK"] = str(rank)
    else:
        raise ValueError(f"Local rank {rank} is already set")


def get_local_rank_optional() -> int | None:
    return _LOCAL_RANK


def get_local_rank() -> int:
    return 0 if _LOCAL_RANK is None else _LOCAL_RANK


def clear_local_rank() -> None:
    global _LOCAL_RANK

    _LOCAL_RANK = None
    os.environ.pop("LOCAL_RANK", None)


def set_world_size(world_size: int) -> None:
    global _WORLD_SIZE

    if world_size != _WORLD_SIZE:
        _WORLD_SIZE = world_size
        os.environ["WORLD_SIZE"] = str(world_size)
    else:
        raise ValueError(f"World size {world_size} is already set")


def get_world_size_optional() -> int | None:
    return _WORLD_SIZE


def get_world_size() -> int:
    return 1 if _WORLD_SIZE is None else _WORLD_SIZE


def clear_world_size() -> None:
    global _WORLD_SIZE

    _WORLD_SIZE = None
    os.environ.pop("WORLD_SIZE", None)


def set_local_world_size(local_world_size: int) -> None:
    global _LOCAL_WORLD_SIZE

    if local_world_size != _LOCAL_WORLD_SIZE:
        _LOCAL_WORLD_SIZE = local_world_size
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
    else:
        raise ValueError(f"World size {local_world_size} is already set")


def get_local_world_size_optional() -> int | None:
    return _LOCAL_WORLD_SIZE


def get_local_world_size() -> int:
    return 1 if _LOCAL_WORLD_SIZE is None else _LOCAL_WORLD_SIZE


def clear_local_world_size() -> None:
    global _LOCAL_WORLD_SIZE

    _LOCAL_WORLD_SIZE = None
    os.environ.pop("LOCAL_WORLD_SIZE", None)


def set_master_addr(master_addr: str) -> None:
    global _MASTER_ADDR

    if master_addr != _MASTER_ADDR:
        os.environ["MASTER_ADDR"] = _MASTER_ADDR = master_addr
    else:
        raise ValueError(f"Master address {master_addr} is already set")


def get_master_addr() -> str:
    assert _MASTER_ADDR is not None, "Master address is not yet set"
    return _MASTER_ADDR


def clear_master_addr() -> None:
    global _MASTER_ADDR

    _MASTER_ADDR = None
    os.environ.pop("MASTER_ADDR", None)


def set_master_port(port: int) -> None:
    global _MASTER_PORT

    if port != _MASTER_PORT:
        _MASTER_PORT = port
        os.environ["MASTER_PORT"] = str(port)
    else:
        raise ValueError(f"Master port {port} is already set")


def get_master_port() -> int:
    assert _MASTER_PORT is not None, "Master port is not yet set"
    return _MASTER_PORT


def clear_master_port() -> None:
    global _MASTER_PORT

    _MASTER_PORT = None
    os.environ.pop("MASTER_PORT", None)


def is_master() -> bool:
    return get_rank() == 0


def is_distributed() -> bool:
    return _INIT_METHOD is not None


def set_init_method(init_method: str) -> None:
    global _INIT_METHOD

    if init_method != _INIT_METHOD:
        os.environ["INIT_METHOD"] = _INIT_METHOD = init_method
    else:
        raise ValueError(f"Init method {init_method} is already set")


def get_init_method() -> str:
    assert _INIT_METHOD is not None, "Init method is not yet set"
    return _INIT_METHOD


def clear_init_method() -> None:
    global _INIT_METHOD

    _INIT_METHOD = None
    os.environ.pop("INIT_METHOD", None)


def set_dist(
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
    master_addr: str,
    master_port: int,
    init_method: str,
) -> None:
    set_rank(rank)
    set_local_rank(local_rank)
    set_world_size(world_size)
    set_local_world_size(local_world_size)
    set_master_addr(master_addr)
    set_master_port(master_port)
    set_init_method(init_method)


def clear_dist() -> None:
    clear_rank()
    clear_local_rank()
    clear_world_size()
    clear_local_world_size()
    clear_master_addr()
    clear_master_port()
    clear_init_method()
