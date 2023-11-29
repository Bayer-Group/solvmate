
all:
	make test_stack
	make test_graph
	make install
	make py_ext_test
	make test_pair_rank

clean:
	rm -rf build
	rm -rf dist
	rm -rf gcyc.egg-info
	rm *.so
	rm *.out

test_stack:
	cc -o stack_test.out stack_test.c && ./stack_test.out

test_graph:
	cc -o graph_test.out graph_test.c && ./graph_test.out

py_ext:
	python setup.py build
	python setup.py install

install:
	make py_ext

py_ext_test:
	make py_ext
	pytest test_gcycmodule.py

test_pair_rank:
	pytest --doctest-modules pair_rank.py

	