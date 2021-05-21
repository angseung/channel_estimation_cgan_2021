function [G] = get_gini_index(p)
    
    N = length(p);
    temp_sum = 0;
    abs_p = abs(p .* (p').');
    l1_p = sum(abs_p);
    [~, p_ind] = sort(abs_p);
    p_asd = abs_p(p_ind);


    for i = 1 : N
        curr_p = p_asd(i);
        temp_sum = temp_sum + (curr_p / l1_p) * ((N - i + 0.5) / N);
    end

    G = 1 - 2 * temp_sum;
    
    % Gini index value MUST be within [0, 1]...
    isValid = (G >= 0) && (G <= 1);
    assert(isValid)
    
end

