% Close all old plots
close all
clear all

% Variables
fontsize = 20;
save_to_file = 1;
linewidth = 3;

%% Our data from testing
% Matrix sizes
matsize = [500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000];

% System Names
names = {'Intel Xeon E3-1505M (single)','Intel Xeon E3-1505M (openmp)','Quadro M2200',...
    'AMD FX-8120 (single)','AMD FX-8120 (openmp)','GeForce GTX 970',...
    'Intel Core i7-7800X (single)','Intel Core i7-7800X (openmp)','GeForce GTX 1080 Ti'};

% Thinkpad p51
cpu1 = [32.4400,47.7600,64.5600,84.0800,105.9600,128.4400,188.4000,257.2800,327.8400,430.2800,520.7200,801.1200,1152.9600,1566.7200,2052.0000,2586.6800,3194.2000];
cpu1dev = [6.0470,6.6229,0.7526,5.0193,2.2888,2.0016,5.4991,2.3069,2.6635,7.9374,10.8500,4.7440,6.2128,11.8272,7.4027,6.6013,12.3321];
omp1 =[7.200,10.160,16.240,22.800,25.560,32.000,43.560,59.080,77.400,95.040,110.600,169.560,241.520,326.200,433.120,533.960,661.320];
omp1dev = [1.357,1.515,4.466,6.119,2.655,3.162,3.226,3.187,3.111,0.824,2.315,1.675,1.269,1.265,19.319,1.969,7.455];
gpu1 = [30.4000,45.9600,62.7200,80.8800,101.2400,125.6000,182.1200,252.2000,320.2400,396.5600,498.1600,746.4000,1068.9200,1451.0000,2992.2000,4091.6800,5643.8000];
gpu1dev = [3.2125,5.6884,6.5270,4.3573,6.8544,3.1875,4.7943,8.1927,4.2359,5.4117,7.1704,11.08513,5.3135,24.4426,30.2139,99.4206,35.6236];


% Desktop #1 AMD and 970
cpu2 = [37.1600,53.2400,73.0800,96.1600,123.1600,151.0000,223.5200,295.2800,385.6800,486.6800,600.2800,937.9600,1404.9200,1885.3600,2458.9600,3128.3600,3794.5200];
cpu2dev = [1.6415,2.1960,2.3481,2.3096,3.0024,3.0332,4.6829,4.9195,4.8143,6.0842,7.2747,8.7383,43.3063,46.1977,51.3945,62.0438,62.6681];
omp2 = [15.600,22.000,28.320,38.400,46.520,58.800,78.560,106.640,137.160,173.520,270.040,414.240,578.840,674.360,867.840,1047.280,1314.400];
omp2dev = [2.728,3.200,2.894,4.391,2.982,4.948,2.531,4.417,4.921,4.365,24.756,28.976,23.349,39.685,49.829,12.969,84.658];
gpu2 = [4.2000,5.2800,7.9200,9.3600,10.6800,13.1200,18.4000,24.5200,32.9200,39.8800,49.9200,76.1200,113.9600,148.8800,220.9200,292.2400,381.0400];
gpu2dev = [0.4000,0.4490,0.4833,0.6248,0.5455,0.8635,0.4899,0.8060,1.6473,0.9086,1.3242,2.0262,3.6494,1.8181,2.3138,2.3712,8.7749];

% Desktop #2 i7 and 1080ti
cpu3 = [21.880,31.520,43.040,56.840,72.440,88.720,126.640,170.560,225.880,289.360,349.080,546.840,785.520,1105.720,1410.440,1774.280,2201.320];
cpu3dev = [4.311,5.449,6.539,8.269,10.100,11.255,15.455,21.526,25.416,33.920,39.395,61.120,86.876,119.127,156.352,193.631,239.381];
omp3 = [21.120,29.880,30.640,31.800,35.760,38.040,45.840,59.800,68.840,80.560,104.360,143.000,220.840,291.520,345.720,434.800,546.160];
omp3dev = [11.944,6.814,5.878,9.831,7.522,7.861,10.578,13.270,11.814,12.589,11.778,20.934,14.195,12.875,42.405,46.163,43.534];
gpu3 = [3.000,3.600,4.800,5.040,6.400,7.360,10.600,13.000,17.160,19.480,24.440,37.160,51.120,69.720,92.240,114.360,139.680];
gpu3dev = [0.000,0.490,0.800,0.196,1.020,1.764,1.095,0.000,1.617,0.574,0.571,1.869,0.325,0.601,3.338,0.480,0.546];


%% Plot the data!
fh1 = figure('name','Convolution Runtime');
set(gcf,'PaperPositionMode','auto')
set(gcf,'defaultuicontrolfontsize',fontsize);
set(gcf,'defaultuicontrolfontname','Bitstream Charter');
set(gcf,'DefaultAxesFontSize',fontsize);
set(gcf,'DefaultAxesFontName','Bitstream Charter');
set(gcf,'DefaultTextFontSize',fontsize);
set(gcf,'DefaultTextFontname','Bitstream Charter');

errorbar(matsize,cpu1,cpu1dev,'b-','LineWidth',linewidth); hold on;
errorbar(matsize,omp1,omp1dev,'r:','LineWidth',linewidth); hold on;
errorbar(matsize,gpu1,gpu1dev,'g--','LineWidth',linewidth); hold on;

errorbar(matsize,cpu2,cpu2dev,'b-','LineWidth',linewidth); hold on;
errorbar(matsize,omp2,omp2dev,'r:','LineWidth',linewidth); hold on;
errorbar(matsize,gpu2,gpu2dev,'g--','LineWidth',linewidth); hold on;

errorbar(matsize,cpu3,cpu3dev,'b-','LineWidth',linewidth); hold on;
errorbar(matsize,omp3,omp3dev,'r:','LineWidth',linewidth); hold on;
errorbar(matsize,gpu3,gpu3dev,'g--','LineWidth',linewidth); hold on;

%========================================================================
%========================================================================

% errorbar(matsize,cpu1,cpu1dev,'-','LineWidth',linewidth); hold on;
% errorbar(matsize,omp1,omp1dev,':','LineWidth',linewidth); hold on;
% errorbar(matsize,gpu1,gpu1dev,'--','LineWidth',linewidth); hold on;
% 
% errorbar(matsize,cpu2,cpu2dev,'-','LineWidth',linewidth); hold on;
% errorbar(matsize,omp2,omp2dev,':','LineWidth',linewidth); hold on;
% errorbar(matsize,gpu2,gpu2dev,'--','LineWidth',linewidth); hold on;
% 
% errorbar(matsize,cpu3,cpu3dev,'-','LineWidth',linewidth); hold on;
% errorbar(matsize,omp3,omp3dev,':','LineWidth',linewidth); hold on;
% errorbar(matsize,gpu3,gpu3dev,'--','LineWidth',linewidth); hold on;


%ylim([0 5])
legend(names,'Location','northwest');
ylabel('Average Time (ms)');
xlabel('Image Width/Height (px)');
set(gcf,'Position',[0 0 1200 600])


%% Save them to file
if save_to_file
    print(fh1,'-dpng','-r500','plot_times_type.png')
end

