


To check for SSE2 support use `cat /proc/cpuinfo` and look in the "flags" section for SSE2.

https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE2&expand=115,5192,5195



### Processor Number 1 (`cat /proc/cpuinfo`)
```
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 158
model name	: Intel(R) Xeon(R) CPU E3-1505M v6 @ 3.00GHz
stepping	: 9
microcode	: 0x5e
cpu MHz     : 3911.132
cache size  : 8192 KB
physical id	: 0
siblings	: 8
core id		: 0
cpu cores	: 4
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch epb intel_pt tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx rdseed adx smap clflushopt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp
bugs		:
bogomips	: 6000.00
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:
```


### GPU Number 1 (`nvidia-smi -i 0 -q`)
```
Timestamp                           : Tue Dec  5 12:42:15 2017
Driver Version                      : 384.90

Attached GPUs                       : 1
GPU 00000000:01:00.0
    Product Name                    : Quadro M2200
    Product Brand                   : Quadro
    Display Mode                    : Enabled
    Display Active                  : Enabled
    Persistence Mode                : Disabled
    Accounting Mode                 : Disabled
    Accounting Mode Buffer Size     : 1920
    Driver Model
        Current                     : N/A
        Pending                     : N/A
    Serial Number                   : N/A
    GPU UUID                        : GPU-97b43966-7dfb-6389-02cf-8a71c943bb0d
    Minor Number                    : 0
    VBIOS Version                   : 84.06.76.00.15
    MultiGPU Board                  : No
    Board ID                        : 0x100
    GPU Part Number                 : N/A
    PCI
        Bus                         : 0x01
        Device                      : 0x00
        Domain                      : 0x0000
        Device Id                   : 0x143610DE
        Bus Id                      : 00000000:01:00.0
        Sub System Id               : 0x225117AA
        GPU Link Info
            PCIe Generation
                Max                 : 3
                Current             : 1
            Link Width
                Max                 : 16x
                Current             : 16x
        Bridge Chip
            Type                    : N/A
            Firmware                : N/A
        Replays since reset         : 0
        Tx Throughput               : 3000 KB/s
        Rx Throughput               : 1000 KB/s
    Fan Speed                       : N/A
    Performance State               : P8
    FB Memory Usage
        Total                       : 4029 MiB
        Used                        : 503 MiB
        Free                        : 3526 MiB
    BAR1 Memory Usage
        Total                       : 256 MiB
        Used                        : 14 MiB
        Free                        : 242 MiB
    Compute Mode                    : Default
    Utilization
        Gpu                         : 29 %
        Memory                      : 11 %
        Encoder                     : 0 %
        Decoder                     : 0 %
    Encoder Stats
        Active Sessions             : 0
        Average FPS                 : 0
        Average Latency             : 0
    Temperature
        GPU Current Temp            : 39 C
        GPU Shutdown Temp           : 0 C
        GPU Slowdown Temp           : 96 C
        GPU Max Operating Temp      : 101 C
        Memory Current Temp         : N/A
        Memory Max Operating Temp   : N/A
    Clocks
        Graphics                    : 135 MHz
        SM                          : 135 MHz
        Memory                      : 405 MHz
        Video                       : 405 MHz
    Applications Clocks
        Graphics                    : 696 MHz
        Memory                      : 2754 MHz
    Default Applications Clocks
        Graphics                    : 694 MHz
        Memory                      : 2754 MHz
    Max Clocks
        Graphics                    : 1038 MHz
        SM                          : 1038 MHz
        Memory                      : 2754 MHz
        Video                       : 851 MHz
```




