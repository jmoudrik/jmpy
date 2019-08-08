.DEFAULT_GOAL := hello

hello:
	@echo 'Hello. Run `make install` for installation or `make clean` for pkg cleanup.'

install:
	@echo 'Installing the `jmpy` module'
	sudo python3 setup.py install
	make clean

clean:
	sudo rm -rf build dist jmpy.egg-info
