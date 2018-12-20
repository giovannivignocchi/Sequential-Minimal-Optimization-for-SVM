path = 'C:\Users\giova\Dropbox\documenti\MATLAB\Sequential Minimal Optimization for SVM\Dataset\Data\Dataset analysis';
saveResult = 1;

% For reproducibility
seed = 100;
rng(seed);

disp('--------------------  diabetes ---------------------')
disp('')
diabetes;
disp('')
disp('--------------------  in_codrna  ---------------------')
disp('')
in_codrna;
disp('')
disp('--------------------  magic  ---------------------')
disp('')
magic;
disp('')
disp('--------------------  ringnorm  ---------------------')
disp('')
ringnorm;
disp('')

% Save the current workspace
if saveResult
    varFile = strcat(path,'\var.m');
    save(varFile,'MdlDiabets','MdlCodrna','MdlMagic','MdlRingnorm');
end