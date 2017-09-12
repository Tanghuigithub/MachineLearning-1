# 语音基础
![](/assets/Direct Matching on Acoustic Level.png)


## MFCC（梅尔频率倒谱系数）
决定尝试新的特征提取方法[MFCC]，和上面那篇论文一样是基于能量分布来提特征，常应用在语音识别。
### Mel scale

The Mel scale relates perceived frequency, or pitch, of a pure tone to its actual measured frequency. Humans are much better at discerning small changes in pitch at low frequencies than they are at high frequencies. Incorporating this scale makes our features match more closely what humans hear.

The formula for converting from frequency to Mel scale is:
$$
M(f)= 1125\ln(1+f/700)
$$

MFCC是浅层特征，只要通过语音本身的分析就能得到，但不同说话人的共有特征还体现在其他特点上，仅通过MFCC是无法捕捉到的。
### 音素

![](/assets/屏幕快照 2017-04-15 16.50.01.png)
![](/assets/屏幕快照 2017-04-15 17.06.32.png)

## Dataset

[Librispeech][10]
  [10]: http://www.danielpovey.com/files/2015_icassp_librispeech.pdf

