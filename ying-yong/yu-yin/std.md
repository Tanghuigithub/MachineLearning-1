# 音频对比技术

## 理论
音频对比主要是从音频信号中提取特征，通过特征进行比对来检索。

参考案例[dejavu](https://github.com/worldveil/dejavu)。

### 特征提取

图中提取的过程就是通过频谱最大值点来建模。

![](http://willdrevo.com/public/img/posts/dejavu-post/spectrogram_peaks.png)

### 特征构建

在完成最大值点的提取后，需要进行特征的构建。

![](http://willdrevo.com/public/img/posts/dejavu-post/spectrogram_zoomed.png)

特征是通过最大值点之间的距离来建模，例如本案例中就采用的是两个最大值点的距离、位置信息作为一个固定的特征作为音频特征信息。

### 检索

有了上述音频特征后，就可以对两个不同音频进行检索，最大相似度的地方就是相似点。需要指出的是，这种技术最适用于录音片段的检索，如若音频内容相同但演绎方式不同（提取的特征点不同）则无法有效识别。

## 实践

这里仍以_dejavu_为例，通过具体代码实现整个流程。详情可参考[Audio Fingerprinting
with Python and Numpy](http://willdrevo.com/fingerprinting-and-audio-recognition-with-python/)。区别于原歌曲检索的应用，这里尝试应用于中文录音的相似音频检索中。

### 环境

- `pyaudio` ：[官网][10]for grabbing audio from microphone
- `ffmpeg`[安装][11]：[官网][12]for converting audio files to .wav format
- `pydub`：a Python ffmpeg wrapper
- `pymysql`：MySQLdb for interfacing with MySQL databases
    - sudo apt-get install mysql-server
    - sudo mysql_secure_installation  
    - systemctl status mysql.service
- `matplotlib`:
    - dateutil 2.0

### 数据预处理
这里使用的是两个人相同文本的录音频谱，可以看到断句不大一样。
完成了分句，提取了单句话不同人的版本，语速、重音都不一样，但频谱图能看出一些关联。![](/assets/10-LuoLing-2016110814.e6fae055bd0a41e28aee16e870c1a80b.pdf)
![频谱对比][13]

#### 句子分割

 首先将长段的句子分割成短句子，这里使用的是 `pydub.AudioSegment()`利用句子之间的短暂安静来分句。因此，首先要做的是归一化音频的强度（amplitude）。

```python
# make audio file to be the same average amplitude 
normalized_sound = match_target_amplitude(sound, -20.0)
# must be silent for at least half a second
min_silence_len=500
# consider it silent if quieter than -16 dBFS
silence_thresh=-40
```

## 提取峰值特征

对于同一个短句，可以看到因为语调、语速的等原因，提取到的特征点并不相似。

![屏幕快照 2017-04-05 10.38.53.png-1092.9kB][9]

这种利用峰值出现的（time，frequency）坐标进行hash的方法对**时间的对齐**应该有一定程度依赖。

> limit：决定是否对整个track进行fingerprint 
DEFAULT_WINDOW_SIZE = 4096 * 16
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_FAN_VALUE = 15

> t[0] = the time (remember, we made a spectrogram? time bins.)
t[1] = the frequency
t[2] = the amplitude

First, we can define a neighborhood around a particular point, and define some point as a peak if it's greater than all the points in its `neighborhood`.

treating the spectrogram as an image and using the **image processing toolkit** and techniques from `scipy` to find peaks. A combination of a `high pass filter` (accentuating high amplitudes) and scipy `local maxima` structs did the trick.

![此处输入图片的描述][15]

> DEFAULT_AMP_MIN = 2
PEAK_NEIGHBORHOOD_SIZE = 10


## Fingerprint hashing

    hash(frequencies of peaks, time difference between peaks) = fingerprint hash value
    
the point is that by taking into account more than a single peak's values you create fingerprints that have more entropy and therefore contain more information. Thus they are more powerful identifiers of songs since they will collide less.

![此处输入图片的描述][16]

> PEAK_NEIGHBORHOOD_SIZE:确定保留峰值的范围

```
if t_delta >= MIN_HASH_TIME_DELTA and t_delta <= MAX_HASH_TIME_DELTA:
    string = "%s|%s|%s" % (freq1, freq2, t_delta)
    # 两个peak（freq1、freq2）加上它们之间的时间差（t_delta）一起进行hash
    h = hashlib.sha1(string.encode('utf-8'))
    yield (h.hexdigest()[0:FINGERPRINT_REDUCTION], t1)
```

## Fingerprint Table

```
CREATE TABLE fingerprints ( 
     hash binary(10) not null,
     song_id mediumint unsigned not null, 
     offset int unsigned not null, 
     INDEX(hash),
     UNIQUE(song_id, offset, hash)
);
```

using a `SHA-1` hash and then cutting it down to **half** its size (just the first 20 characters)
> char(40) => char(20) goes from 40 bytes to 20 bytes

Next we'll take this hex encoding and convert it to binary, once again cutting the space down considerably:

> char(20) => binary(10) goes from 20 bytes to 10 bytes

##　Fingerprint Alignment

Actually, we'd be out of luck anyway in the case the playback speed was different, since this would affect the frequency of the playback and therefore the peaks in the spectrogram. At any rate, the playback speed assumption is a good (and important) one.
Under this assumption, for each match we calculate a difference between the offsets:

> difference = database offset from original track - sample offset from recording



## MFCC

### librosa

```
# y : np.ndarray [shape=(n,) or (2, n)]
# sr : number > 0 [scalar]
y, sr = librosa.load(filename)
```
```
[[ 6.94920832  9.25765064  7.58265376  7.82286846  7.4833739   8.37554084
   7.20683539  6.34957351  5.97452396  5.97520647  5.31247496  4.62040589
   4.00647279  3.37785973  3.72667181  4.57756396  3.7387241   2.07888545
   3.72327256  3.05818003  2.68919455  3.16364766  3.5318312   4.07098374
   4.58482052  8.72341577]
 [ 6.85802729  7.27963887  8.91559927  9.41918724  8.10439295  7.58168654
   7.10264537  6.51303901  5.81004773  5.3536778   4.91483512  3.30361871
   2.58184007  3.90461755  4.27312772  3.48394424  3.35036472  2.78788674
   3.37679743  3.23690964  3.14844955  3.33108009  3.25924761  3.56542889
   5.07370861  8.70717444]]
```

```
[[  3.08089496   2.91483664   4.60622414   5.27551393   5.89466681
    5.48047788   5.15521198   4.90173113   6.14954334   6.12586276
    6.84213627   7.38162055   9.91335338   8.86053729   7.51911678
    8.76846366  11.14892981  11.25777967   9.55176643   8.53260828
    6.95944749   5.17796703   7.6529262    9.74778256   9.90562711
    7.24426934]
 [  3.57938634   5.12181324   5.95152519   6.09056898   5.03852239
    5.32969122   6.21627432   5.70506468   6.54646627   6.08395624
    6.09936515   8.03006857  10.2092765    9.32365044   9.12307024
   11.71444812  12.83974206  11.26004138  10.1694855    8.99768156
    7.76897913   5.47010741   8.56368116  10.17972659  11.32790302
    7.89659767]]
```

```
('a06e50dde734765caa4a', 44) ('64fbfc4dfb46553e1bc1', 151) ('76ce0d3be42664a8d53e', 310) ('1176fde2898b2b3e53b6', 310) ('e387e2b01460992f14d3', 406) ('171d31342a2a37ca31a4', 499)
```

```
('f95a45cd0bd96b5abc9b', 35) ('94f6a77c4b36a894510b', 132) ('5da1566bef6dc1033c0c', 132) ('a0b8893e1c98ba9dc611', 244) ('122abcae408f2bed9630', 244) ('bafb67ab005a4cb6004e', 321) ('cd34f4a9030669c2b7e1', 413) ('dc80319805492ffd6a20', 573)
```

```
Normalized distance between the origin sound and segment/man/food2-cut1-chunk0.wav: 75.200960
Normalized distance between the origin sound and segment/man/food2-cut1-chunk1.wav: 93.479585
Normalized distance between the origin sound and segment/man/food2-cut1-chunk10.wav: 88.445749
Normalized distance between the origin sound and segment/man/food2-cut1-chunk11.wav: 99.642177
Normalized distance between the origin sound and segment/man/food2-cut1-chunk12.wav: 86.241258
Normalized distance between the origin sound and segment/man/food2-cut1-chunk2.wav: 127.442735
Normalized distance between the origin sound and segment/man/food2-cut1-chunk3.wav: 83.719548
Normalized distance between the origin sound and segment/man/food2-cut1-chunk4.wav: 90.148167
Normalized distance between the origin sound and segment/man/food2-cut1-chunk5.wav: 85.962117
Normalized distance between the origin sound and segment/man/food2-cut1-chunk6.wav: 85.368065
Normalized distance between the origin sound and segment/man/food2-cut1-chunk7.wav: 86.333645
Normalized distance between the origin sound and segment/man/food2-cut1-chunk8.wav: 87.480143
Normalized distance between the origin sound and segment/man/food2-cut1-chunk9.wav: 88.853638
```

```
Normalized distance between the origin sound and segment/man/food2-cut1-chunk0.wav: 81.403371
Normalized distance between the origin sound and segment/man/food2-cut1-chunk1.wav: 95.513704
Normalized distance between the origin sound and segment/man/food2-cut1-chunk10.wav: 85.688754
Normalized distance between the origin sound and segment/man/food2-cut1-chunk11.wav: 109.291830
Normalized distance between the origin sound and segment/man/food2-cut1-chunk12.wav: 82.123877
Normalized distance between the origin sound and segment/man/food2-cut1-chunk2.wav: 120.010228
Normalized distance between the origin sound and segment/man/food2-cut1-chunk3.wav: 85.776484
Normalized distance between the origin sound and segment/man/food2-cut1-chunk4.wav: 92.703972
Normalized distance between the origin sound and segment/man/food2-cut1-chunk5.wav: 84.796863
Normalized distance between the origin sound and segment/man/food2-cut1-chunk6.wav: 87.864336
Normalized distance between the origin sound and segment/man/food2-cut1-chunk7.wav: 92.112338
Normalized distance between the origin sound and segment/man/food2-cut1-chunk8.wav: 88.700106
Normalized distance between the origin sound and segment/man/food2-cut1-chunk9.wav: 71.750063
```
## 其他研究
论文_Robust Audio Hashing for Content Identification_

- 先做framing，傅里叶变换得到频谱。
- 划分frequency band（300-3000Hz），分别计算energy，由相邻band的energy之间的关系计算一个bit。共32个band，于是得到32bit的Hash值。


 ![屏幕快照 2017-09-08 10.50.19.png-1400.4kB][14]

[9]: http://static.zybuluo.com/sixijinling/uz2bu1hah0pj29rhzuo4x8uf/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-04-05%2010.38.53.png

[10]: http://people.csail.mit.edu/hubert/pyaudio/
[11]: http://noalgo.info/874.html
[12]: http://ffmpeg.org/
[13]: http://static.zybuluo.com/sixijinling/dqrsl1v3drulfhte0y5zvjb2/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-04-03%2011.38.21.png
[14]: http://static.zybuluo.com/sixijinling/hznjeb0158ymxrs1h3803wgm/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-09-08%2010.50.19.png