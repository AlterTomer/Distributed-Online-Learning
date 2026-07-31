# Thin alias over tasks.py, which is the single source of truth for what each
# target does. `make` is unavailable on the Windows dev machine; this file exists
# for CI and for anyone on a POSIX box who reaches for `make` by reflex.

PYTHON ?= python

.PHONY: install test test-fast lint format typecheck clean \
        reference x0 x1 x1b x2 x3 x4 x5 x6 sweep figures list

list:
	@$(PYTHON) tasks.py --list

install test test-fast lint format typecheck clean reference x0 x1 x1b x2 x3 x4 x5 x6 sweep figures:
	@$(PYTHON) tasks.py $@
