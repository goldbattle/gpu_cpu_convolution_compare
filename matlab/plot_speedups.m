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

% Thinkpad p51
cpu1 = [32.4400,47.7600,64.5600,84.0800,105.9600,128.4400,188.4000,257.2800,327.8400,430.2800,520.7200,801.1200,1152.9600,1566.7200,2052.0000,2586.6800,3194.2000];
omp1 =[7.200,10.160,16.240,22.800,25.560,32.000,43.560,59.080,77.400,95.040,110.600,169.560,241.520,326.200,433.120,533.960,661.320];
gpu1 = [30.4000,45.9600,62.7200,80.8800,101.2400,125.6000,182.1200,252.2000,320.2400,396.5600,498.1600,746.4000,1068.9200,1451.0000,2992.2000,4091.6800,5643.8000];

% Desktop #1 AMD and 970
cpu2 = [37.1600,53.2400,73.0800,96.1600,123.1600,151.0000,223.5200,295.2800,385.6800,486.6800,600.2800,937.9600,1404.9200,1885.3600,2458.9600,3128.3600,3794.5200];
omp2 = [15.600,22.000,28.320,38.400,46.520,58.800,78.560,106.640,137.160,173.520,270.040,414.240,578.840,674.360,867.840,1047.280,1314.400];
gpu2 = [4.2000,5.2800,7.9200,9.3600,10.6800,13.1200,18.4000,24.5200,32.9200,39.8800,49.9200,76.1200,113.9600,148.8800,220.9200,292.2400,381.0400];

% Desktop #2 i7 and 1080ti
cpu3 = [21.880,31.520,43.040,56.840,72.440,88.720,126.640,170.560,225.880,289.360,349.080,546.840,785.520,1105.720,1410.440,1774.280,2201.320];
omp3 = [21.120,29.880,30.640,31.800,35.760,38.040,45.840,59.800,68.840,80.560,104.360,143.000,220.840,291.520,345.720,434.800,546.160];
gpu3 = [3.000,3.600,4.800,5.040,6.400,7.360,10.600,13.000,17.160,19.480,24.440,37.160,51.120,69.720,92.240,114.360,139.680];

%% Calculate speed up relative to the AMD CPU

% single vs multi cpu
omp1_rel_cpu1 = cpu1./omp1;
omp2_rel_cpu2 = cpu2./omp2;
omp3_rel_cpu3 = cpu3./omp3;

% gpu vs multi (skip crappy quadro)
gpu2_rel_omp1 = omp1./gpu2;
gpu2_rel_omp2 = omp2./gpu2;
gpu2_rel_omp3 = omp3./gpu2;
gpu3_rel_omp1 = omp1./gpu3;
gpu3_rel_omp2 = omp2./gpu3;
gpu3_rel_omp3 = omp3./gpu3;



%% Plot #2
fh1 = figure('name','Speedup CPUs');
set(gcf,'PaperPositionMode','auto')
set(gcf,'defaultuicontrolfontsize',fontsize);
set(gcf,'defaultuicontrolfontname','Bitstream Charter');
set(gcf,'DefaultAxesFontSize',fontsize);
set(gcf,'DefaultAxesFontName','Bitstream Charter');
set(gcf,'DefaultTextFontSize',fontsize);
set(gcf,'DefaultTextFontname','Bitstream Charter');

plot(matsize,omp1_rel_cpu1,'r-','LineWidth',linewidth); hold on;
plot(matsize,omp2_rel_cpu2,'b-','LineWidth',linewidth); hold on;
plot(matsize,omp3_rel_cpu3,'g','LineWidth',linewidth); hold on;

%ylim([0 5])
names = {'Intel Xeon E3-1505M (openmp) w.r.t Intel Xeon E3-1505M (single)',...
    'Intel Core i7-7800X (openmp) w.r.t Intel Core i7-7800X (single)',...
    'AMD FX-8120 (openmp) w.r.t AMD FX-8120 (single)'
    };
legend(names,'Location','southeast');%,'northwest');
ylabel('Average Speedup Fraction');
xlabel('Image Width/Height (px)');
set(gcf,'Position',[0 0 1200 600])

%% Plot #2
fh2 = figure('name','Speedup GPUs');
set(gcf,'PaperPositionMode','auto')
set(gcf,'defaultuicontrolfontsize',fontsize);
set(gcf,'defaultuicontrolfontname','Bitstream Charter');
set(gcf,'DefaultAxesFontSize',fontsize);
set(gcf,'DefaultAxesFontName','Bitstream Charter');
set(gcf,'DefaultTextFontSize',fontsize);
set(gcf,'DefaultTextFontname','Bitstream Charter');

%plot(matsize,gpu1_rel_omp1,'r-','LineWidth',linewidth); hold on;
plot(matsize,gpu2_rel_omp1,'b-','LineWidth',linewidth); hold on;
plot(matsize,gpu2_rel_omp2,'r-','LineWidth',linewidth); hold on;
plot(matsize,gpu2_rel_omp3,'g-','LineWidth',linewidth); hold on;
plot(matsize,gpu3_rel_omp1,'b:','LineWidth',linewidth); hold on;
plot(matsize,gpu3_rel_omp2,'r:','LineWidth',linewidth); hold on;
plot(matsize,gpu3_rel_omp3,'g:','LineWidth',linewidth); hold on;


%ylim([0 5])
names = {
    'GeForce GTX 970 w.r.t Intel Xeon E3-1505M (openmp)',...
    'GeForce GTX 970 w.r.t AMD FX-8120 (openmp)',...
    'GeForce GTX 970 w.r.t Intel Core i7-7800X (openmp)',...
    'GeForce GTX 1080 Ti w.r.t Intel Xeon E3-1505M (openmp)',...
    'GeForce GTX 1080 Ti w.r.t AMD FX-8120 (openmp)',...
    'GeForce GTX 1080 Ti w.r.t Intel Core i7-7800X (openmp)',...
    };
legend(names,'Location','northeast');%,'northwest');
ylabel('Average Speedup Fraction');
xlabel('Image Width/Height (px)');
set(gcf,'Position',[0 0 1200 600])



%% Save them to file
if save_to_file
    print(fh1,'-dpng','-r500','plot_speedup_cpus.png')
    print(fh2,'-dpng','-r500','plot_speedup_gpus.png')
end

