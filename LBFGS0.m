function [xj, fval, it] = LBFGS0(fg, x0, lmax, scale, progTol, optTol, ...
                                 max_iter, report_iter, varargin)
%
%
% S. Ulbrich, May 2002
% Modified by Truyen, Oct 2004.

% Limited memory BFGS-method with Powell-Wolfe stepsize rule. 
%
% Input:  
%   fg      : name of a matlab-function [f,g] = fg(x)
%               that returns value and gradient
%               of the objective function depending on the
%               number of the given ouput arguments
%   x0      :   starting point (ROW VECTOR)
%   lmax    : maximal number of stored updates for
%                   the limited memory BFGS-approximation
%                   if not given, lmax = 60 is used
%   opt     : other options for the function fhd(X,opt)
%   scale   : initial step length, to avoid numerical problem
%   epsilon : stopping criteria
%   max_iter  :   number of iterations
% Output: 
%   sols    : series of solutions
%   fvals   : series of loglikehood values per iteration
%   it      : number of iterations toward convergence
%   timex   : timex taken

if report_iter > 0
    fprintf('%10s %15s\n', 'Iteration', 'Func Val');    	
end

% constants 0<del<theta<1, del<1/2 for Wolfe condition
del = 0.001;
theta = 0.9;
% constant 0<al<1 for sufficient decrease condition
al = 0.001;
%lmax = 60;
f_len = length(x0);

P = zeros(f_len,lmax);
D = zeros(f_len,lmax);
ga = zeros(lmax,1);
rho = zeros(lmax,1);
l = 0;
gak = 1;

xj = x0;
% [f,g] = vector_convert(fg,xj,opt);
[f, g] = fg(xj, varargin{:});
fval = f;
if max(abs(g)) <= optTol
    return;
end
nmg0 = norm(g);
nmg = nmg0;
it = 0;
l = 0;
ln = 1;

stp = zeros(1,f_len);
% main loop
while it < max_iter
    %sig = r;
    % compute BFGS-step s = B*g;
    q = g;
    for j = 1:l
        i = mod(ln-j-1,lmax) + 1;
        ga(i) = rho(i)*(P(:,i)'*q);
        q = q-ga(i)*D(:,i);
    end
    r = gak*q;
    
    for j = l:-1:1
        i = mod(ln-j-1,lmax) + 1;
        be = rho(i)*(D(:,i)'*r);
        r = r + (ga(i)-be)*P(:,i);
    end
    s = r;
    step = 'LBFGS';
    
    % check if BFGS-step provides sufficient decrease; else take gradient
    stg = s'*g;
    if stg < min(al,nmg)*nmg*norm(s)
        s = g;
        stg = s'*g;
        step = 'Grad';
    end
    
    % choose sig by Powell-Wolfe stepsize rule
    [xn,fn,gn] = wolfe(xj,s,stg,fg,f,g,del,theta,scale,varargin{:});
    
     
    % update BFGS-matrix
    d = g - gn;
    p = xj - xn;
    dtp = d'*p;
    if dtp >= 1e-8*norm(d)*norm(p)
        rho(ln) = 1/dtp;
        D(:,ln) = d;
        P(:,ln) = p;
        l = min(l + 1,lmax);
        ln = mod(ln,lmax) + 1;
        if l == lmax
            gak = dtp/(d'*d);
        end
    end
    xj = xn;
    g = gn;
    f = fn;
    nmg = norm(g);

    it = it + 1;
    
    prev_fval = fval;
    fval = f;

    if mod(it, report_iter) == 0
        fprintf('%10d %15.5e\n', it, f);
    end
    
    if isnan(f) %it is numerically unstable for some reason, Truyen 20th May, 2005
        break;
    end

    
    if it > 1 && prev_fval ~= 0 && abs(fval/prev_fval -1) < progTol
%     if abs(fval-prev_fval) < progTol
        break; %stop the optimiser
    end
    if max(abs(g)) < optTol
        break;
    end
end

end

%----------------------------------------------------------------
function [best_x,best_f,best_g] = wolfe(xj,s,stg,fct,f,g,del,theta,scale,varargin)
%
%
% S. Ulbrich, May 2002
% Modified:
%   Truyen, Oct 2004 to control the step size, 
%       avoiding ill-conditioning and unsuccessful step size search
%   Truyen, Nov 2005 to use best solution found so far

% This code comes with no guarantee or warranty of any kind.
%
% function [sig,xn,fn] = wolfe(xj,s,stg,fct,f,del,scale,theta)
%
% Determines stepsize satisfying the Powell-Wolfe conditions
%
% Input:  xj       current point
%         s        search direction (xn = xj-sig*s)
%         stg      stg = s'*g
%         fct      name of a matlab-function [f] = fct(x)
%                  that returns the value of the objective function
%         f        current objective function value f = fct(xj)
%         g        current gradient
%         del      constant 0<del<1/2 in Armijo condition f-fn> = sig*del*stg
%         theta    constant del<theta<1 in Wolfe condition gn'*s< = theta*stg
%         scale     initial stepsize (usually scale = 1)
%
% Output: sig      stepsize sig satisfying the Armijo condition
%         xn       new point xn = xj-sig*s
%         fn       fn = f(xn)
%
max_s = max(abs(s));
sig1 = scale/max_s;
sig = sig1;
xn = xj - sig*s;

best_x = xj;
best_f = f;
best_g = g;


% [fn,gn] = vector_convert(fct,xn,opt);
[fn, gn] = fct(xn, varargin{:});
% Determine maximal sig = scale/2^k satisfying Armijo
count = 0;
while (count < 5 && f-fn < del*sig*stg)
	count = count + 1;
    sig = 0.5*sig;
    xn = xj-sig*s;
%     [fn] = vector_convert(fct,xn,opt);
    fn = fct(xn, varargin{:});
end
if sig~=sig1
%     [fn,gn] = vector_convert(fct,xn,opt);
    [fn, gn] = fct(xn, varargin{:});
end

if fn < best_f
    best_x = xn;
    best_f = fn;
    best_g = gn;
end

% If sig = scale satisfies Armijo then try sig = 2^k*scale
% until sig satisfies also the Wolfe condition
% or until sigp = 2^(k + 1)*scale violates the Armijo condition
if (sig == sig1)
    xnn = xj - 2*sig*s;
%     [fnn,gnn] = vector_convert(fct,xnn,opt);
    [fnn, gnn] = fct(xnn, varargin{:});
    count = 0;
    while count < 5 & (gn'*s > theta*stg) & (f - fnn >= 2*del*sig*stg)
        count = count + 1;
        sig = 2*sig;
        xn = xnn;
        fn = fnn;
        gn = gnn;
        xnn = xj - 2*sig*s;
%         [fnn,gnn] = vector_convert(fct,xnn,opt);
        [fnn, gnn] = fct(xnn, varargin{:});

        if fnn < best_f
            best_x = xnn;
            best_f = fnn;
            best_g = gnn;
        end
    end
end
sigp = 2*sig;

% Perform bisektion until sig satisfies also the Wolfe condition
count = 0;
while count < 5 && (gn'*s > theta*stg)
    count = count + 1;
    sigb = 0.5*(sig + sigp);
    xb = xj - sigb*s;
%     [fnn,gnn] = vector_convert(fct,xb, opt);
    [fnn, gnn] = fct(xb, varargin{:});
    if fnn < best_f
        best_x = xb;
        best_f = fnn;
        best_g = gnn;
    end
    if (f - fnn >= del*sigb*stg)
        sig = sigb;
        xn = xb;
        fn = fnn;
        gn = gnn;
    else
        sigp = sigb;
    end
end

end