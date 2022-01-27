# Lottery_ticket_to_predict
预测代码  
基于seq2seq加入注意力机制的彩票预测代码  
现是应用在湖北十一选五上  
每次预测五个数,平均能对三个左右,最高五个,需要根据实际 走势 定杀号 或是 做 周期倍投  
5月份开始做的,5月份合计盈利7895元  
有兴趣的可以联系我,我们一起完善代码,发财致富  
无论你是 走势高手,还是 代码高手 还是 有兴趣都可以联系我 我需要人入伙 各玩各的  
不收钱 不交钱  
一起就行  


联系方式 先加QQ聊，聊的开心，有戏，我们再加微信。
transformer_translate 测试 结果如下
最近160期 测试结果如下  默认任二 为三个数  成本  为6元   默认三 为四个数  成本  为8元   
2 [0, -6, 0, 0, -6, -6, 0, 0, 12, 12, -6, -6, -6, 0, -6, 0, -6, 0, -6, 0, -6, 12, 12, 0, 0, 0, 0, 0, 0, 0, 12, 12, -6, 0, -6, 0, -6, 12, -6, 0, 0, 0, 0, 0, -6, -6, 0, 0, 12, 0, 12, 0, 0, 12, 12, 0, -6, -6, 0, 12, 12, 0, 0, 0, 12, 0, -6, -6, 12, 0, 0, -6, 0, 12, 12, 0, 12, 12, 0, 12, -6, -6, 12, 12, 0, 12, 0, 0, 0, 0, 12, 0, 0, 0, -6, 12, -6, -6, 0, -6, 0, -6, 0, 0, -6, -6, 0, 0, 0, -6, 0, 0, 0, -6, 0, -6, 12, 0, 0, -6, -6, -6, -6, 0, -6, 12, 12, -6, 0, 0, 0, -6, 0, -6, 0, 0, 12, -6, -6, 0, 0, 0, 0, 0, -6, 0, 0, -6, -6, 12, -6, -6, 0, 12, -6]
任二 一共投注 158 期 最终结果 输赢值 为 78  
3 [-8, -8, 11, -8, -8, -8, 11, -8, 68, 11, -8, -8, -8, -8, -8, -8, -8, -8, -8, 11, -8, 11, 68, -8, 11, 11, -8, -8, 11, 11, 68, 68, -8, -8, -8, -8, -8, 68, -8, -8, 11, -8, -8, -8, -8, -8, -8, -8, 11, -8, 11, -8, -8, 11, 11, -8, -8, -8, 11, 11, 11, -8, 11, 11, 11, 11, -8, -8, 68, -8, 11, -8, -8, 11, 11, -8, 11, 68, -8, 11, -8, -8, 11, 11, 11, 11, 11, -8, 68, -8, 11, 68, -8, -8, -8, -8, 11, -8, -8, 11, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, 11, 11, 11, -8, 11, -8, 68, 11, 11, -8, -8, -8, -8, -8, -8, 11, 68, -8, 11, -8, 11, -8, -8, -8, 11, 11, 11, -8, -8, -8, 11, -8, -8, -8, -8, 11, 11, -8, -8, 11, -8, -8, 11, 68, -8]
任三 一共投注 158 期 最终结果 输赢值 为 633  
QQ号如下：2304699441  


刚刚看这已经是两年前的demo了,前几天有人提issues问我,不然我都快忘记这个仓了,   
然后反思了一下这个代码,存在着一些错点和可以改进的地方,   
所有的前提是**彩票是有规律可寻的,或者说是有细微规律的**  
如果是随机的,万法皆空,还不如机选。   
改进如下：   
1、现在都是借用文本的seq2seq，或者rnn网络作为底层网络，默认为同期的号码是有关联的，其实应该不然，同期出的号不应该有关联，不然第一个号码预测错误，之后所有的都将白费。因为数据集也么有按照出号顺序制作。改进方式：换用其他网络。
2、加入注意力机制，前期的号可以对预测号进行约束。
3、加入数据底层公式,万变不离其宗，需要根据公式来做底层依据，不能全靠空空的神经网络。
                            添加于2022年1月27日
另今日给自己卜卦，卦象为 离为火卦 ；卦辞：占此卦者遇天宫，富禄必然降人间。一切谋望皆吉庆，愁闲消散主平安。祈祷2022年，必将有福禄，所求皆可得。
