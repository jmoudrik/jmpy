.DEFAULT_GOAL := hello

hello:
	@echo 'Hello. Run `make install` for installation or `make clean` for pkg cleanup.'

install:
	@echo 'Installing the `jmpy` module'
	sudo python3 setup.py install
	make clean

test:
	python3 -m doctest -v jmpy/utils.py

clean:
	sudo rm -rf build dist jmpy.egg-info
