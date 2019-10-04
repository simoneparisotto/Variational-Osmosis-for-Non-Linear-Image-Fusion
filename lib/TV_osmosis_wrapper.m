function [u,v,E] = TV_osmosis_wrapper(foreground,background,alpha,params)

% initialisation of u0,v0
switch params.flag_initialisation
    case 0
        u = foreground;
    case 1
        u = alpha.*foreground + (1-alpha).*background;
    case 2
        mf = alpha.*foreground;     mf = sum(sum(mf))./sum(sum(alpha));
        mb = (1-alpha).*background; mb = sum(sum(mb))./sum(sum(1-alpha));
        u = alpha.*bsxfun(@times,mf,ones(size(foreground))) + (1-alpha).*bsxfun(@times,mb,ones(size(background)));
end
u0 = u;
v0 = (foreground.^alpha) .* (background.^(1-alpha));

fprintf(params.fileID,'-------------------------------------------------------------------------\n');
fprintf('-------------------------------------------------------------------------\n')

% single cycle - block iPiano with backtracking
[u,v,L1,L2,E] = block_ipiano(foreground,background,alpha,u0,v0,params);

params.L1 = L1; % keep track of the already computed L
params.L2 = L2; % keep track of the already computed L

if params.plot_figures
    figure(1),
    subplot(2,2,1)
    imshow(u-params.offset,[0,1])
    title('u')
    subplot(2,2,2)
    imshow(v-params.offset,[0,1])
    title('v')
    subplot(2,2,3)
    imshow(u-u0,[])
    title('u-u0')
    pause(1)
end

return