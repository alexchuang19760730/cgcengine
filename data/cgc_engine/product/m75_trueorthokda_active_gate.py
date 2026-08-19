try:
    from .m75_trueorthokda_active_runtime import (
        main,
        run_m75_trueorthokda_active_gate,
        run_m75_trueorthokda_active_runtime,
    )
except ImportError:
    from m75_trueorthokda_active_runtime import (  # type: ignore
        main,
        run_m75_trueorthokda_active_gate,
        run_m75_trueorthokda_active_runtime,
    )

__all__ = [
    "run_m75_trueorthokda_active_runtime",
    "run_m75_trueorthokda_active_gate",
    "main",
]


if __name__ == "__main__":
    main()
