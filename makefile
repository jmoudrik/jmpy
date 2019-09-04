project := jmpy
.DEFAULT_GOAL := hello

hello:
	@echo 'Hello, this is makefile for $(project).'
	@echo 'Run `make install` for installation or `make clean` for pkg cleanup.'

install:
	@echo 'Installing the `$(project)` module'
	sudo pip3 install .
	make clean

uninstall:
	@echo 'Un-installing the `$(project)` module'
	sudo pip3 uninstall $(project)

test:
	python3 -m doctest -v $(project)/utils.py

clean:
	sudo rm -rf build dist $(project).egg-info

turn:
	make clean
	yes | make uninstall
	make install
