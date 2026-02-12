%%%sample rotate image
function GIF_movie_demo(x,type,alpha_value,titstr,initial_azi,initial_el)
% Create sample 3D scatter data
%rng(0);
%n = 500;
%X = randn(n,1);
%Y = randn(n,1);
%Z = randn(n,1) + 0.5*X; % correlated for nicer structure
%c = sqrt(X.^2+Y.^2+Z.^2);

% Create figure and initial scatter3
fig = figure('Color','w','Position',[200 200 700 600]);
ax = axes('Parent',fig);
% ss(Idir,J)=scatter3(data.x_tsne(Itype,1), data.x_tsne(Itype,2), data.x_tsne(Itype,3), 3,type(Itype),'filled');

s = scatter3(ax, x(:,1), x(:,2),x(:,3), 3, type, 'filled');
s.MarkerEdgeAlpha=alpha_value;
s.MarkerFaceAlpha=alpha_value;
      
colormap(jet)
colorbar
axis equal
xlabel('T-SNE Dimension 1');
ylabel('T-SNE Dimension 2');
zlabel('T-SNE Dimension 3');
title(titstr);

% Lighting and view
%view(45,25)
view(initial_azi,initial_el);
grid on
xlim([-2 2]);ylim([-2 2]);zlim([-2 2]);

drawnow

% Parameters for rotation and output
nFrames = 120;         % number of frames in full rotation
az0 = initial_azi;              % starting azimuth
el =initial_el;               % elevation (fixed)
outputGIF = true;      % set false to skip GIF creation
gifName = 'rotating_scatter.gif';
delayTime = 8*0.03;      % delay between frames in seconds

% Preallocate capture
frames(nFrames) = struct('cdata',[],'colormap',[]);

% Rotate by changing azimuth angle

axis vis3d
for k = 1:nFrames
    az = bnorm(az0 + 360*(k-1)/nFrames);
    %view(ax, az, el)

    camorbit(360/nFrames,0)
    title(sprintf('%s az: %6.2f el:%6.2f',titstr,az,el))
    drawnow
    frames(k) = getframe(fig);
    if outputGIF
        img = frame2im(frames(k));
        [A,map] = rgb2ind(img,256);
        if k == 1
            imwrite(A,map,gifName,'gif','LoopCount',Inf,'DelayTime',delayTime);
        else
            imwrite(A,map,gifName,'gif','WriteMode','append','DelayTime',delayTime);
        end
    end
end

% If not writing GIF, play movie once
if ~outputGIF
    movie(fig, frames, 1, 1/delayTime)
end

% Close or leave figure open as desired (comment/uncomment)
% close(fig)