g++ -O0 -g -I/usr/local//Cellar/openblas/0.2.14_1/include -std=c++11 -Wall -o mxnet-all.o -c mxnet-all.cc 
ar rcs mxnet-all.a mxnet-all.o
