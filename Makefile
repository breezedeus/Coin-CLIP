package:
	rm -rf build
	python setup.py sdist bdist_wheel

VERSION := $(shell sed -n "s/^__version__ = '\(.*\)'/\1/p" coin_clip/__version__.py)
upload:
	python -m twine upload  dist/coin_clip-$(VERSION)* --verbose

.PHONY: package upload
