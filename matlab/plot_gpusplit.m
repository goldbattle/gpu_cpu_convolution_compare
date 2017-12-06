% Close all old plots
close all
clear all

% Variables
fontsize = 20;
save_to_file = 0;
linewidth = 3;

%% Our data from testing
% Matrix sizes
matsize = [500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000];

% System Names
names = {'Quadro M2200','GeForce GTX 970','GeForce GTX 1080 Ti'};
parts = {'Copy Host to Device','Kernel Execution','Copy Device to Host','Memory Free'};

% Thinkpad p51 (quadro M2200)
gpu1 = [30.4000,45.9600,62.7200,80.8800,101.2400,125.6000,182.1200,252.2000,320.2400,396.5600,498.1600,746.4000,1068.9200,1451.0000,2992.2000,4091.6800,5643.8000];
copyto1 = [0.040,1.000,1.000,1.120,2.000,2.040,3.160,4.800,6.200,8.600,10.160,11.0851,22.560,30.760,40.280,51.200,65.600];
kernel1 = [27.440,42.400,58.120,75.160,94.280,117.360,170.880,237.920,301.560,373.680,470.600,705.400,1012.320,1375.280,2892.960,3967.320,5490.800];
copyback1 = [0.120,0.320,1.040,1.120,2.000,2.360,3.240,4.440,6.440,8.000,9.400,15.480,22.600,30.760,40.040,50.320,62.560];
free1 = [0.120,0.160,1.120,1.120,1.240,1.800,2.280,3.120,4.160,4.840,6.000,7.440,7.680,7.520,7.520,7.680,7.520];

% Desktop #1 (970)
gpu2 = [4.2000,5.2800,7.9200,9.3600,10.6800,13.1200,18.4000,24.5200,32.9200,39.8800,49.9200,76.1200,113.9600,148.8800,220.9200,292.2400,381.0400];
copyto2 = [0.000,0.280,1.000,1.080,1.800,2.000,3.160,4.440,6.280,7.520,8.920,14.240,20.000,27.760,35.880,45.720,55.840];
kernel2 = [2.000,2.240,4.040,4.600,5.360,7.040,10.000,13.520,18.840,22.840,30.120,45.880,71.880,91.880,147.520,198.800,268.680];
copyback2 = [0.000,0.040,1.000,1.080,1.000,1.560,2.480,3.480,4.800,5.960,7.160,10.640,15.480,21.320,27.880,35.800,43.640];
free2 = [0.000,0.000,0.000,0.080,0.000,0.000,1.000,1.000,1.000,1.000,2.000,3.080,4.040,5.200,7.000,9.000,10.000];

% Desktop #2 (1080ti)
gpu3 = [3.000,3.600,4.800,5.040,6.400,7.360,10.600,13.000,17.160,19.480,24.440,37.160,51.120,69.720,92.240,114.360,139.680];
copyto3 = [0.000,0.000,0.000,0.120,1.000,1.000,2.000,2.000,3.000,4.000,5.000,8.040,11.960,16.040,21.040,26.280,32.840];
kernel3 = [0.000,1.000,1.320,2.000,2.000,2.000,4.440,5.960,8.040,9.000,11.000,17.760,24.000,34.000,46.120,57.000,69.000];
copyback3 = [0.000,0.000,0.000,0.000,1.000,1.000,1.080,2.000,3.000,4.000,4.720,7.000,11.000,15.000,19.000,24.640,30.160];
free3 = [0.080,0.120,0.280,0.000,0.240,0.040,0.480,0.360,1.360,1.000,1.040,1.320,2.000,2.160,3.040,4.000,5.000];


%% Plot  #1
fh1 = figure('name','Quadro M2200 Split');
set(gcf,'PaperPositionMode','auto')
set(gcf,'defaultuicontrolfontsize',fontsize);
set(gcf,'defaultuicontrolfontname','Bitstream Charter');
set(gcf,'DefaultAxesFontSize',fontsize);
set(gcf,'DefaultAxesFontName','Bitstream Charter');
set(gcf,'DefaultTextFontSize',fontsize);
set(gcf,'DefaultTextFontname','Bitstream Charter');

plot(matsize,gpu1,'--','LineWidth',linewidth); hold on;
area(matsize',[copyto1;kernel1;copyback1;free1]'); hold on;

%ylim([0 5])
legend([names{1},parts],'Location','northwest');
ylabel('Average Time (ms)');
xlabel('Image Width/Height (px)');
set(gcf,'Position',[0 0 1200 600])

%% Plot  #2
fh1 = figure('name','GTX 970 Split');
set(gcf,'PaperPositionMode','auto')
set(gcf,'defaultuicontrolfontsize',fontsize);
set(gcf,'defaultuicontrolfontname','Bitstream Charter');
set(gcf,'DefaultAxesFontSize',fontsize);
set(gcf,'DefaultAxesFontName','Bitstream Charter');
set(gcf,'DefaultTextFontSize',fontsize);
set(gcf,'DefaultTextFontname','Bitstream Charter');

plot(matsize,gpu2,'--','LineWidth',linewidth); hold on;
area(matsize',[copyto2;kernel2;copyback2;free2]'); hold on;

%ylim([0 5])
legend([names{2},parts],'Location','northwest');
ylabel('Average Time (ms)');
xlabel('Image Width/Height (px)');
set(gcf,'Position',[0 0 1200 600])

%% Plot #3
fh3 = figure('name','GTX 1080ti Split');
set(gcf,'PaperPositionMode','auto')
set(gcf,'defaultuicontrolfontsize',fontsize);
set(gcf,'defaultuicontrolfontname','Bitstream Charter');
set(gcf,'DefaultAxesFontSize',fontsize);
set(gcf,'DefaultAxesFontName','Bitstream Charter');
set(gcf,'DefaultTextFontSize',fontsize);
set(gcf,'DefaultTextFontname','Bitstream Charter');

plot(matsize,gpu3,'--','LineWidth',linewidth); hold on;
area(matsize',[copyto3;kernel3;copyback3;free3]'); hold on;

%ylim([0 5])
legend([names{3},parts],'Location','northwest');
ylabel('Average Time (ms)');
xlabel('Image Width/Height (px)');
set(gcf,'Position',[0 0 1200 600])


%% Save them to file
if save_to_file
    print(fh1,'-dpng','-r500','plot_gpusplit_1_M2200.png')
    print(fh2,'-dpng','-r500','plot_gpusplit_2_GTX970.png')
    print(fh3,'-dpng','-r500','plot_gpusplit_3_GTX1080ti.png')
end

