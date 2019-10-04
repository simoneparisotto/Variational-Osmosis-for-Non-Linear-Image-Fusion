% Variational Osmosis for Non-Linear Image Fusion
% Copyright: Simone Parisotto, L. Calatroni, A. Bugeau, N. Papadakis and C.-B. Schönlieb
% Date: 04/10/2019

clear
close all
clc

addpath ./lib
addpath ./dataset/

EXPERIMENT = [1]; % here 1 = puppets, 4 = facefusion

%% Parameters
params.plot_figures = 0;
params.flag_verbose = 1;

% for the model
ETA     = [0.5];     % [5 10 100]; % stack of weights for TV
MU      = [100];     % [5 10 100]; % stack of weights for fidelity v
GAMMA   = [1];       % [1 10 100]; % stack of weights for fidelity u
EPSILON = [0.05];    % HUBER-TV
% for iPiano (Alg. 4)
params.lambda1 = 2;   % backtracking update
params.lambda2 = 1.2; % backtracking update
params.beta1   = 0.4;
params.beta2   = 0.4;
params.L1      = 1.0; % estimated starting Lipschitz
params.L2      = 1.0; % estimated starting Lipschitz
params.beta1   = 0.4; % inertial parameter
params.beta2   = 0.4; % inertial parameter
% for blurring alpha-map
FLAG_BLUR      = [0]; % 0 or 1 or [0,1]
BLUR           = [5]; % [0.5,1.5,3,15]

% Initialisation
% 0: (u0 = f), 1: (u0 = alpha*f + (1-alpha)*b ), 2: AVG
FLAG_initialisation = [0];

% iterations
params.N = 10000; % maxiter iPiano
params.T = 10000; % maxiter primal-dual nested in iPiano

% tolerances
params.tol_ipiano      = 1e-6;
params.tol_primal_dual = 1e-4;

%% START OF THE ALGORITHM
for experiment = EXPERIMENT % which data are you using?
    
    for flag_blur = FLAG_BLUR % do you wish to blur the alpha map?
        for blur_test = 1:(flag_blur * (numel(BLUR)-1) + 1) % how many blur test do you want?
            
            for gamma = GAMMA % which parameter for fidelity wrt u?
                for mu = MU % which fidelity term for the field?
                    
                    OSM     = [1]; % weights for osmosis: here fixed but it can be a stack
                    for osm = OSM  % which fidelity term for osmosis?
                        
                        for eta = ETA % which fidelity term for TV?
                            for epsilon = EPSILON % Huber TV?
                                
                                for flag_initialisation = FLAG_initialisation % which initalisation do you prefer?
                                    
                                    % create parameters fot the model
                                    params.eta                 = eta;
                                    params.osm                 = osm;
                                    params.mu                  = mu;
                                    params.gamma               = gamma;
                                    params.epsilon             = epsilon;
                                    params.flag_blur           = flag_blur;
                                    params.blur_test           = blur_test;
                                    params.flag_initialisation = flag_initialisation;
                                    params.experiment          = experiment;
                                    
                                    % create filename
                                    filename_u = ['./results/',num2str(experiment),'_output_u','_init',num2str(params.flag_initialisation),'_alphablur',num2str(params.flag_blur*BLUR(blur_test)),'_eta',num2str(params.eta),'_mu',num2str(params.mu),'_gamma',num2str(params.gamma),'_eps',num2str(params.epsilon)];
                                    filename_v = ['./results/',num2str(experiment),'_output_v','_init',num2str(params.flag_initialisation),'_alphablur',num2str(params.flag_blur*BLUR(blur_test)),'_eta',num2str(params.eta),'_mu',num2str(params.mu),'_gamma',num2str(params.gamma),'_eps',num2str(params.epsilon)];
                                    filename_txt = [filename_u,'_',datestr(now),'.txt'];
                                    params.fileID = fopen(filename_txt,'w');
                                    
                                    % load images
                                    foreground = im2double(imread([num2str(experiment),'_foreground.png']));
                                    background = im2double(imread([num2str(experiment),'_background.png']));
                                    alpha      = mat2gray(im2double(imread([num2str(experiment),'_alpha.png'])));
                                    if params.flag_blur
                                        alpha = imgaussfilt(double(alpha==1),BLUR(blur_test));
                                    end
                                    if max(size(foreground,1),size(foreground,2))>1500 && experiment == 2
                                        foreground = imresize_old(foreground,0.4);
                                        background = imresize_old(background,0.4);
                                        alpha      = imresize_old(alpha,0.4);
                                    end
                                    C = size(foreground,3);
                                    
                                    ufinal = zeros(size(foreground));
                                    vfinal = zeros(size(foreground));
                                    E      = NaN(params.N,size(foreground,3));
                                    
                                    fprintf(params.fileID,'\n########################\n');
                                    fprintf(params.fileID,'###### TV-OSMOSIS ######\n');
                                    fprintf(params.fileID,'########################\n\n');
                                    
                                    fprintf(params.fileID,'experiment: %d\n',experiment);
                                    
                                    if params.flag_blur
                                        fprintf(params.fileID,'alpha blur: %f\n',BLUR(blur_test));
                                    else
                                        fprintf(params.fileID,'alpha blur: no\n');
                                    end
                                    
                                    switch params.flag_initialisation
                                        case 0
                                            fprintf(params.fileID,'initial.  : u0 = f\n');
                                        case 1
                                            fprintf(params.fileID,'initial.  : u0 = alpha*f + (1-alpha)*b\n');
                                        case 2
                                            fprintf(params.fileID,'initial.  : u0 = alpha*avg(f) + (1-alpha)*avg(b)\n');
                                    end
                                    
                                    fprintf(params.fileID,'\n###### PARAMETERS ######\n');
                                    fprintf(params.fileID,'-- model --\n');
                                    fprintf(params.fileID,'eta             : %2.2f\n',params.eta);
                                    fprintf(params.fileID,'mu              : %2.2f\n',params.mu);
                                    fprintf(params.fileID,'gamma           : %2.2f\n',params.gamma);
                                    fprintf(params.fileID,'epsilon (huber) : %2.2f\n',params.epsilon);
                                    fprintf(params.fileID,'-- iPiano --\n');
                                    fprintf(params.fileID,'beta1           : %2.2f\n',params.beta1);
                                    fprintf(params.fileID,'beta2           : %2.2f\n',params.beta2);
                                    fprintf(params.fileID,'lambda1         : %2.2f\n',params.lambda1);
                                    fprintf(params.fileID,'lambda2         : %2.2f\n',params.lambda2);
                                    fprintf(params.fileID,'L1 (starting)   : %2.2f\n',params.L1);
                                    fprintf(params.fileID,'L2 (starting)   : %2.2f\n',params.L2);
                                    fprintf(params.fileID,'tol iPiano      : %2.2e\n',params.tol_ipiano);
                                    fprintf(params.fileID,'maxiter         : %03d\n',params.N);
                                    fprintf(params.fileID,'-- Primal-Dual --\n');
                                    fprintf(params.fileID,'tol PD          : %2.2e\n',params.tol_primal_dual);
                                    fprintf(params.fileID,'maxiter PD      : %03d\n',params.T);
                                    
                                    fprintf('\n########################\n');
                                    fprintf('###### TV-OSMOSIS ######\n');
                                    fprintf('########################\n\n');
                                    
                                    fprintf('experiment: %d\n',experiment);
                                    
                                    if params.flag_blur
                                        fprintf('alpha blur: %f\n',BLUR(blur_test));
                                    else
                                        fprintf('alpha blur: no\n');
                                    end
                                    
                                    switch params.flag_initialisation
                                        case 0
                                            fprintf('initial.  : u0 = f\n');
                                        case 1
                                            fprintf('initial.  : u0 = alpha*f + (1-alpha)*b\n');
                                        case 2
                                            fprintf('initial.  : u0 = alpha*avg(f) + (1-alpha)*avg(b)\n');
                                    end
                                    
                                    fprintf('\n###### PARAMETERS ######\n');
                                    fprintf('-- model --\n');
                                    fprintf('eta             : %2.2f\n',params.eta);
                                    fprintf('mu              : %2.2f\n',params.mu);
                                    fprintf('gamma           : %2.2f\n',params.gamma);
                                    fprintf('epsilon (huber) : %2.2f\n',params.epsilon);
                                    fprintf('-- iPiano --\n');
                                    fprintf('beta1           : %2.2f\n',params.beta1);
                                    fprintf('beta2           : %2.2f\n',params.beta2);
                                    fprintf('lambda1         : %2.2f\n',params.lambda1);
                                    fprintf('lambda2         : %2.2f\n',params.lambda2);
                                    fprintf('L1 (starting)   : %2.2f\n',params.L1);
                                    fprintf('L2 (starting)   : %2.2f\n',params.L2);
                                    fprintf('tol iPiano      : %2.2e\n',params.tol_ipiano);
                                    fprintf('maxiter         : %03d\n',params.N);
                                    fprintf('-- Primal-Dual --\n');
                                    fprintf('tol PD          : %2.2e\n',params.tol_primal_dual);
                                    fprintf('maxiter PD      : %03d\n',params.T);
                                    
                                    % add offset for positivity
                                    params.offset = 1;
                                    foreground    = foreground + params.offset;
                                    background    = background + params.offset;
                                    
                                    % CORE ALGORITHM (each colour channel is processed separately)
                                    t_start = tic;
                                    for c = 1:C
                                        params.c = c;
                                        [ufinal(:,:,params.c),vfinal(:,:,params.c),E(:,params.c)] = TV_osmosis_wrapper(foreground(:,:,c),background(:,:,c),alpha(:,:,c),params);
                                    end
                                    t_end = toc(t_start);
                                    
                                    % remove offset
                                    foreground = foreground-params.offset;
                                    background = background-params.offset;
                                    ufinal     = ufinal-params.offset;
                                    vfinal     = vfinal-params.offset;
                                    
                                    fprintf(params.fileID,'\ncputime: %f s.',t_end);
                                    fclose(params.fileID);
                                    
                                    imwrite(ufinal,[filename_u,'_time',num2str(t_end),'.png']);
                                    imwrite(vfinal,[filename_v,'_time',num2str(t_end),'.png']);
                                    
                                    if params.plot_figures
                                        figure,
                                        subplot(1,3,1)
                                        imshow(foreground,[0,1])
                                        title('foreground')
                                        subplot(1,3,2)
                                        imshow(background,[0,1])
                                        title('background')
                                        subplot(1,3,3)
                                        imshow(ufinal,[0,1])
                                        title('result')
                                    end
                                    
                                    clc
                                    
                                    % save results in .mat format
                                    save([filename_u,'.mat']);
                                    
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end