function [u,v,L1,L2,E] = block_ipiano(f,b,alpha,u,v,params)

eta             = params.eta;   % for the regulariser
osm             = params.osm;   % for osmosis
mu              = params.mu;    % for fidelity on v
gamma           = params.gamma; % for fidelity on u
lambda1         = params.lambda1;
lambda2         = params.lambda2;
beta1           = params.beta1;
beta2           = params.beta2;
L1              = params.L1;
L2              = params.L2;
N               = params.N;
T               = params.T;
tol_ipiano      = params.tol_ipiano;
tol_primal_dual = params.tol_primal_dual;
flag_verbose    = params.flag_verbose;

[mm,nn] = size(u);

[~,D1,D2]      = grad_forward(u);
options.niter  = T;
options.tol    = tol_primal_dual;

% inline functions for primal-dual
K      = @(v)       reshape(cat(2,D1*v(:),D2*v(:)),mm,nn,2);
KS     = @(y)       reshape(D1.'*reshape(y(:,:,1),mm*nn,1) + D2.'*reshape(y(:,:,2),mm*nn,1),mm,nn);
ProxFS = @(y,sigma) (y./(1+sigma*params.epsilon))./repmat(max(1,norms(y./(1+sigma*params.epsilon),2,3)/eta),1,1,size(y,3));   

GRAD   = @(u) cat(3,reshape(D1*u(:),mm,nn),reshape(D2*u(:),mm,nn));
DIV    = @(u) reshape(D1.'*reshape(u(:,:,1),[],1)+D2.'*reshape(u(:,:,2),[],1),mm,nn);

% inline functions for iPiano
g1     = @(v)   (eta)     .* huber(GRAD(v),params.epsilon);
g2     = @(u,v) (osm/2)   .* v.*norms(GRAD(u./v),2,3).^2;
g3     = @(v)   (mu/2)    .* norms(v-(f.^alpha).*(b.^(1-alpha)),2,3).^2;
g4     = @(u)   (gamma/2) .* norms(sqrt(alpha).*(u-f),2,3).^2;

pixel_energy = @(u,v) g1(v) + g2(u,v) + g3(v) + g4(u);
total_energy = @(u,v) sum(sum( pixel_energy(u,v) ));

Ouv    = @(u,v) g2(u,v) + g3(v);

% DIV here is the adjoint of GRAD which is "-div" so it changes sign here
dvO_1  = @(u,v) - sum( GRAD(u).*GRAD(u),3 ) ./ (v.^2);
dvO_2  = @(u,v) +(4*u./v.^3)  .* sum( GRAD(u).*GRAD(v),3);
dvO_3  = @(u,v) - DIV( 2*(u./v.^2)  .* GRAD(u) );
dvO_4  = @(u,v) - 3*u.^2./v.^4 .* sum(GRAD(v).^2,3);
dvO_5  = @(u,v) +v.*DIV( (u.^2./v.^3) .* GRAD(v) );

grad_duO = @(A,u,v) (osm./v).*reshape( A*u(:) ,mm,nn);
grad_dvO = @(u,v)   (osm/2) * (dvO_1(u,v)+dvO_2(u,v)+dvO_3(u,v)+dvO_4(u,v)+dvO_5(u,v)) + mu*(v-(f.^alpha).*(b.^(1-alpha)));

test_u = @(p1,p2,A,u,v,L) sum(sum( Ouv(p1,v) - Ouv(u,v) - grad_duO(A,u,v).*(p1-u) - (L/2).*norms(p1-u,2,3).^2 - 1e-10 ));
test_v = @(p1,p2,u,v,L)   sum(sum( Ouv(u,p2) - Ouv(u,v) - grad_dvO(u,v)  .*(p2-v) - (L/2).*norms(p2-v,2,3).^2 - 1e-10 ));

% iPiano with backtracking
u_old = u;
v_old = v;
E     = NaN(N,1);

if flag_verbose
    message1 = sprintf('  ITER   | (   L1    |    L2   ) |   energy   |   diff   |  PD Res (iter) \n');
    fprintf(params.fileID,message1);
    fprintf(message1)
    fprintf(params.fileID,'-------------------------------------------------------------------------\n');
    fprintf('-------------------------------------------------------------------------\n')
end

for n = 1:N
    
    flag = 1;
    
    A  = osmosis_discretization(v);
    grad_u = grad_duO(A,u,v);
    grad_v = grad_dvO(u,v);
    
    flag_compute_u = 1;
    flag_compute_v = 1;
    
    while flag
        
        % initialisation
        if beta1 > 0.5
            an1         = 1.99*(1-beta1)/L1;
        else
            an1         = 0.99*(1-2*beta1)/L1;
        end
        if beta2 > 0.5
            an2         = 1.99*(1-beta2)/L2;
        else
            an2         = 0.99*(1-2*beta2)/L2;
        end
        
        % explicit step of iPiano
        if flag_compute_u
            ud = u - an1.*grad_u + beta1*(u-u_old);
        end
        if flag_compute_v
            vd = v - an2.*grad_v + beta2*(v-v_old);
        end
        
        % primal-dual for the proximal step of iPiano
        ProxGu = @(q,an)  (gamma.*alpha.*f + ud/an) ./ (gamma.*alpha + 1/an);
        ProxGv = @(q,tau) (vd/an2 + q/tau) ./ (1/an2 + 1/tau);
        
        if flag_compute_u
            p1            = ProxGu(ud,an1);
        end
        if flag_compute_v
                [p2,res,iter] = primal_dual(vd, 1/an2, K, KS, ProxFS, ProxGv, options);
        end
        
        gap1 = test_u(p1,v,A,u,v,L1);
        gap2 = test_v(u,p2,u,v,L2);
        
        if gap1<0 && flag_compute_u
            u_old = u;
            u     = p1;
            L1    = L1/lambda2;
            flag_compute_u = 0;
        else
            if flag_compute_u
                L1 = lambda1*L1;
            end
        end
        
        if gap2<0 && flag_compute_v
            v_old = v;
            v     = p2;
            L2 = L2/lambda2;
            flag_compute_v = 0;
        else
            if flag_compute_v
                L2 = lambda1*L2;
            end
        end   
        
        if flag_compute_u==0 && flag_compute_v==0
            E(n)        = total_energy(u,v); 
            energy_diff = abs(total_energy(u,v)-total_energy(u_old,v_old))./abs(total_energy(u_old,v_old));
            flag  = 0;
            if flag_verbose
                message2 = sprintf('   %03d   | (%2.2e | %2.2e) | %2.4e | %2.2e | %2.2e (%d)\n',n,lambda2*L1,lambda2*L2,E(n),energy_diff,res,iter);
                fprintf(params.fileID,message2);
                fprintf(message2);
            end
        end
       
    end
    
    if (energy_diff<tol_ipiano && n>=5)
        break
    end
end

return

function z = huber(y,epsilon)
t = norms(y,2,3);
idx = t<epsilon;
z = t - epsilon/2;
z(idx) = (t(idx).^2)./(2*epsilon);
return

% function x = bicgstab_wrapper(A,b,tol,iter)
%         [x,FLAG,RELRES,ITER] = bicgstab(A,b,tol,iter);
% return