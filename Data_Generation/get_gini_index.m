function [G] = get_gini_index(p)

    N = length(p);
    temp_sum = 0;
    abs_p = abs(p);
    l1_p = sum(abs_p);
    [~, p_ind] = sort(abs_p);
    p_asd = abs(p(p_ind));


    for i = 1 : N
        curr_p = p_asd(i);
        temp_sum = temp_sum + (curr_p / l1_p) * ((N - i + 0.5) / N);
    end 

    G = 1 - 2 * temp_sum;
    
end

