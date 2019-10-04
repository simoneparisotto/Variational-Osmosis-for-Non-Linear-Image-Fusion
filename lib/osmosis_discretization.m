%% Authors: Simone Parisotto, Marco Caliari


function A = osmosis_discretization(v)
% Input:
%   umat  = shadowed image
%   umask = shadow boundary indicator function

[mx, my, ~] = size(v);

x  = linspace(1,mx,mx)';
y  = linspace(1,my,my)';
hx = (max(x)-min(x))/(mx-1);
hy = (max(y)-min(y))/(my-1);

[~,D1classic,D2classic] = grad_forward(ones(mx,my));

% LAPLACIAN
Dxx_old = D1classic.'*D1classic;
Dyy_old = D2classic.'*D2classic;

% MATRIX FOR -div(du)
% average upper (u_{i+1,j} + u_{ij})/2  and lower (u_{ij}+u_{i-1,j})/2
m1xup  = spdiags(ones(mx,2)/2,[0,1],mx,mx);
m1xlow = spdiags(ones(mx,2)/2,[-1,0],mx,mx);
% average upper (u_{i,j+1} + u_{ij})/2  and lower (u_{ij}+u_{i,j-1})/2
m1yup  = spdiags(ones(my,2)/2,[0,1],my,my);
m1ylow = spdiags(ones(my,2)/2,[-1,0],my,my);

M1xup  = kron(speye(my),m1xup);
M1xlow = kron(speye(my),m1xlow);
M1yup  = kron(m1yup,speye(mx));
M1ylow = kron(m1ylow,speye(mx));

% STANDARD DRIFT VECTOR FIELD d
d1ij = zeros(mx+1,my); 
d2ij = zeros(mx,my+1);

d1ij(2:mx,:) = diff(v,1,1)./(v(2:mx,:)+v(1:mx-1,:))*2/hx;
d2ij(:,2:my) = diff(v,1,2)./(v(:,2:my)+v(:,1:my-1))*2/hy;

%  STANDARD OSMOSIS FILTER
Ax = Dxx_old + 1/hx*( ...
    spdiags(reshape(d1ij(2:mx+1,:),mx*my,1),0,mx*my,mx*my)*M1xup - ...
    spdiags(reshape(d1ij(1:mx,:),mx*my,1),0,mx*my,mx*my)*M1xlow);

Ay = Dyy_old + 1/hy*( ...
    spdiags(reshape(d2ij(:,2:my+1),mx*my,1),0,mx*my,mx*my)*M1yup - ...
    spdiags(reshape(d2ij(:,1:my),mx*my,1),0,mx*my,mx*my)*M1ylow);

A = Ax + Ay;

end