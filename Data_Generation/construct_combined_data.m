clear; close all; clc

a = load("Gan_Data/dat_3_2_25g.mat");
b = load("Gan_Data/dat_3_20_25g.mat");

sparse_train_input = a.input_da;
sparse_test_input = a.input_da_test;
sparse_train_output = a.output_da;
sparse_test_output = a.output_da_test;

scatter_train_input = b.input_da;
scatter_test_input = b.input_da_test;
scatter_train_output = b.output_da;
scatter_test_output = b.output_da_test;


input_da = cat(1,sparse_train_input, scatter_train_input);
input_da_test = cat(1,sparse_test_input, scatter_test_input);
output_da = cat(1,sparse_train_output, scatter_train_output);
output_da_test = cat(1,sparse_test_output, scatter_test_output);

save("Gan_Data/comb_dat_2_20.mat",'input_da','output_da','input_da_test','output_da_test','-v7.3')