function kernel_type = prompt_kernel(kernel_list)
    fprintf('=============================\n');
    fprintf('Select a kernel type:\n');
    fprintf('=============================\n');
    for i = 1:length(kernel_list)
        fprintf('  %d → %s\n', i, kernel_list{i});
    end
    fprintf('=============================\n');
    
    kernel_choice = input('Enter the number of the kernel type you want to use: ');
    if kernel_choice < 1 || kernel_choice > length(kernel_list)
        error('Invalid kernel selection. Please enter a number from 1 to %d.\n', length(kernel_list));
    end
    
    kernel_type = kernel_list{kernel_choice};
    fprintf('\n  You selected: [%d] → %s\n\n', kernel_choice, kernel_type);
end
