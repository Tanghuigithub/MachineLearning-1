# 语音基础
![](/assets/Direct Matching on Acoustic Level.png)


## MFCC（梅尔频率倒谱系数）
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


  [10]: http://people.csail.mit.edu/hubert/pyaudio/
  [11]: http://noalgo.info/874.html
  [12]: http://ffmpeg.org/
  [13]: http://static.zybuluo.com/sixijinling/dqrsl1v3drulfhte0y5zvjb2/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-04-03%2011.38.21.png
