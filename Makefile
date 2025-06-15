.PHONY: documentation

documentation:
	uv run --group docs python -m sphinx -M html docs/source docs/build

