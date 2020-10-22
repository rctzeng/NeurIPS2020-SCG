%**************************************************************************
%********  Finding Gangs in War from Signed Networks, KDD2016  ************
%********     Author: Lingyang Chu, Simon Fraser University    ************
%********            Email:  chulingyang@hotmail.com           ************
%********                 All Rights Preserved                 ************
%**************************************************************************
function [X_enumKOCG_cell, time] = enumKOCG(A, B, NG, alpha, beta)
%% This is the main function to enumerate all KOCGs.
%% Note that in this version, all matrix and vector variables should stored in sparse.
% A:        First network source, i.e. the affinity matrix of an undirected graph
% B:        Second network source, i.e. the conflict matrix of another undirected graph
%           If B == [], then A is a signed network, we have to split them up.
% NG:       max number of groups, which is the #col of X (i.e., n-by-NG matrix)
% alpha:    the parameter on the conflict component
% beta:     the parameter on the orthogonal penalty component
% X:        the multi-source grouping result.
% time:     Effective clean running time cost.

%% code as follow
%% Initialization

fprintf('=====================  enumKOCG Start ==================== \n');

% tic toc time flies
T_start = clock;

fprintf('FOCG running ...\n');
% define some global const variable
global shrink_tao conv_thres near_zero K;
shrink_tao = 1e-5; % default = 1e-8
conv_thres = 2e-2; % default = 1e-5
near_zero = 1e-10; % default = 1e-10
K = 200; % the number limit of expanded new vertexes in expansion phase

% check and deal with signed network
if (isempty(B))
    % deal with signed network
    fprintf('Signed network input. Do splitting ... \n');
    B = -A;
    A(A < 0) = 0;
    B(B < 0) = 0;
end

% ensure A and B are symmetric SPARSE matrix of the same size
assert(sum(sum(A - A')) == 0); 
assert(sum(sum(B - B')) == 0);
assert(sum(abs(size(A)-size(B))) == 0);
assert(issparse(A) && issparse(B));

% NV: number of vertexes in both A and B
NV = size(A,1);

% time log
T = [];

% prepare constant matrix H
fprintf('Initialize constant matrices ...\n');
tic;
I = sparse(1:NV, 1:NV, ones(NV,1));
H = alpha*B - beta*I; % H = (alpha*B - beta*I) % big
U = ones(NG,NG) - eye(NG); % U = E - I, for calc of comp_trace
T = [T toc];
fprintf('Constant matrices initialized! Time: %f \n', T(end));    

% init X
fprintf('Initialize X with optimial seeding ... \n');
tic;
X = IOCG(B, NG);
T = [T toc];
fprintf('Initialization complete! Time: %f \n', T(end));

% start FOCG
fprintf('Start FOCG ... \n');
tic;
X_enumKOCG_cell = enumKOCGCore(A, B, H, U, NG);
T = [T toc];
fprintf('All complete !!! Time: %f \n Total Time: %f \n', T(end), sum(T));

% tic toc time flies
time = etime(clock, T_start);

fprintf('===========  enumKOCG Complete! Time: %.2f =========== \n', time);



%% ************************************************************************
%% ***************************  UTILITY FUNCTIONS  ************************
%% ************************************************************************


function X_enumKOCG_cell = enumKOCGCore(A, B, H, U, NG)
%% This function enumerates all cores using the peel-off strategy
%% code as follow

NV = size(A, 1);
glb_mask = logical(ones(NV, 1)); % global_mask: =1 vertex remain, =0 vertex peeled off

counter = 1;

while (sum(glb_mask ~= 0))
    
    % show progress
    fprintf('EnumKOCG progres ... Counter: %d   Remaining vertexes: %d / %d \n', counter, sum(glb_mask), NV);
    
    % init X_loc
    X_loc = IOCG(B(glb_mask, glb_mask), NG); % index of X_loc is local
    
    % check validity
    if (isempty(X_loc))
        return;
    end
    
    % find optimal X_loc 
    [X_loc, FX_Val_log] = FOCGCoreOuter(A(glb_mask, glb_mask), H(glb_mask, glb_mask), X_loc, U, NG);
    
    % clean X_loc
    X_loc(X_loc < 1e-7) = 0;    
    
    % transform X_loc to X_glb
    X_glb = loc2glbX(X_loc, glb_mask);
    
    % log X_glb
    X_enumKOCG_cell{counter, 1} = X_glb;
    counter = counter + 1;
    
    % update glb_mask
    glb_mask(sum(X_glb, 2) > 0) = 0;
    
end



function X_glb = loc2glbX(X_loc, glb_mask)
%% This functio transforms local X to global X
%% code as follow

% first, generate the transform mat
trans_mat = diag(sparse(glb_mask));
trans_mat = trans_mat(:, sum(trans_mat, 1) == 1);

% transform X_loc to X_glb
X_glb = trans_mat*X_loc;


function [X, FX_Val_log] = FOCGCoreOuter(A, H, X, U, NG)
% initialize the global const variables
global conv_thres;

% prepare Mi for i = 1
Xjs = X(:, 2:NG);
Mi = sum(H*Xjs, 2); % Mi when i = 1

%% Start iteration
FX_Val_log  = -1;
outer_iter = 0;
outer_iter_thres = 50; % this threshold is to avoid dead loop caused by dirty data. This won't happen when data is clean.
old_i = 1; % old_i can only be placed outside the while loop
while(outer_iter <= outer_iter_thres)
    outer_iter = outer_iter + 1;
    
    % inner loop
    for i = 1:NG
        % update Mi by delta
        delta = H*(X(:,old_i) - X(:,i));
        Mi = Mi + delta;
        
        % do FOCGInner
        [Xi, fiVal] = FOCGCoreInner(A, X(:,i), Mi);
        X(:,i) = Xi;
        
        % update old_i for later update of Mi
        old_i = i;
    end
    
    % log FX_Val
    cur_FX_Val = calcFXVal(A, H, X, U);
    FX_Val_log = [FX_Val_log cur_FX_Val];
    
    % check out
    if (abs(FX_Val_log(end) - FX_Val_log(end - 1)) <= conv_thres)
        break;
    end
    
    % show info
    fprintf('Outer_iter: %d \t FX_Val: %f \n', outer_iter, FX_Val_log(end));
    
end



function [Xi, fiVal] = FOCGCoreInner(A, Xi, Mi)
%% this is the core function of FOCG, which optimize upon the i-th row of X
%% code as follow
% initialize the global const variables
global conv_thres;

% start iteration
old_fiVal = 0;
counter_thres = 100; % this threshold is to avoid dead loop caused by dirty data. This won't happen when data is clean.
counter = 0;
while(true)
    % do expansion
    Xi = update(Xi, A, Mi);
    
    % do shrinkage
    [Xi, fiVal] = locate(Xi, A, Mi);
    
    % checkout
    counter = counter + 1;
    if (abs(fiVal - old_fiVal) <= conv_thres)
        break;
    elseif (counter >= counter_thres)
        fprintf('Warning: Unnormal break in FOCGCoreInner !!! Iter limited of %d exceeded.\n', counter_thres);
        break;
    else
        old_fiVal = fiVal;
    end
end

fiVal = full(fiVal);



function [newXi, fiVal] = locate(Xi, A, Mi)
% initialize the global const variables
global shrink_tao near_zero;

% check out if there is not enough elements in <X>
if sum(Xi>0) < 2
    newXi = Xi;
    fiVal = 0;
    return;
end

% make the computations local
NV = length(Xi);
sptXi = find(Xi>0);

Xi = Xi(sptXi); 
A  = A(sptXi, sptXi);
Mi = Mi(sptXi);
e  = ones(length(sptXi),1);
Qi = A + e*Mi' + Mi*e';

% start iteration
old_XQiX = 0;
XQiX = 0;
while(true)
    
    % calculate vector AXi
    QiXi = Qi*Xi;
    XQiX = Xi'*QiXi;
        
    % check out
    dens_chg = abs(XQiX - old_XQiX);
    if (dens_chg < shrink_tao || XQiX == 0)
        break;
    else
        old_XQiX = XQiX;
    end
    
    % this is a safe threshold in case we have extremly bad data.
    if (XQiX == 0)
        break;
    end
    
    % update Xi
    Xi = (Xi.*QiXi)/XQiX;
    
        % apply near_zero strategy
    Xi(Xi < near_zero) = 0;
    Xi = Xi/sum(Xi);
end

% recover and return result, density is already assigned
newXi = sparse(NV,1);
newXi(sptXi) = Xi;

fiVal = XQiX + 2*Xi'*Mi;



function newXi = update(Xi, A, Mi)
%% this is the update phase
%% code as follow
% initialize gloabl const variables
global K;

% calculate vector R(Xi) = AXi + Mi
RXi = A*Xi + Mi;

% calculate scalar g(Xi) = Xi'*R(Xi)
gXi = Xi'*RXi;

% calculate vector gamma
gamma = RXi - gXi;
gamma(gamma <= 0) = 0;
gamma(Xi > 0) = 0; % adjust the offset due to numerical error. Essure Z and sigma does not overlap.

% control the number of newly expanded vertexes
[tmp, indi_srt] = sort(gamma, 'descend');
gamma(indi_srt((K+1):end)) = 0;

% check out if gamma is zero vec, which means Z is empty.
if (sum(gamma) == 0)
    newXi = Xi;
    return;
end

% calculate scalar s
s = sum(gamma);

% calcualte scalar eta
eta = gamma'*gamma;

% calculate vector b
b = gamma - s*Xi;

% calculate b'Ab
b_spt = (b ~= 0);
btAb = b(b_spt)'*A(b_spt, b_spt)*b(b_spt);

if (btAb >= 0)
    t = 1/s;
else
    t = min(1/s, -eta/btAb);
end

newXi = Xi + t*b;


function X = IOCG(B, NG)
%% this function initialize X using the k-means++ random seeding method
% NV: number of vertexes in both A and B
NV = size(B,1);

% check validity
if (NV < NG)
    X = [];
    return;
end

% first, randomly pick a point as the first seed, use the distribution of node degree
BNN_vec  = sum(B,2);
BNN2_vec = BNN_vec.*BNN_vec + 1e-10; % Adding 1e-10 to make sure no connection vertex can also be selected under extreme cases.
P_vec  = BNN2_vec/sum(BNN2_vec);
seeds  = rndSample(P_vec);

% second pick the others
for i = 2:NG
    D_vec = mean(B(:, seeds), 2); % average distance from centers
    D_vec(seeds) = 0; % guarantee not repeat seed
    D2_vec = D_vec.*D_vec + 1e-10; % element wise squared. Adding 1e-10 to make sure no connection vertex can also be selected.
    P_vec = D2_vec/sum(D2_vec); % the probability vec
    
    % sample the i-th seed, using P_vec
    seedi = rndSample(P_vec);
    seeds = [seeds seedi];
end

% initialize X
X = sparse(NV, NG);
for i = 1:NG
    X(seeds(i), i) = 1;
end



function seed = rndSample(P_vec)
rd = rand();

sum = 0;
idx = find(P_vec>0);
for i = 1:length(idx)
    
    sum = sum + P_vec(idx(i));
    
    if (sum >= rd)
        break;
    end
end

seed = idx(i);


function FX_Val = calcFXVal(A, H, X, U)
% this function calculates the value of original objective
FX_Val = trace(X'*A*X) + trace(X'*H*X*U);
