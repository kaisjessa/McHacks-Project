import crepe
import math
from scipy.io import wavfile

sr, audio = wavfile.read('sample.wav')
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=False, step_size=100, verbose=0)

print(list(frequency))
print(list(confidence))
#frequency = [0.0, 430.0878947474183, 660.2564602921241, 537.3046543192776, 656.5773825438457, 84.64864566723952, 231.45130293476302, 268.95270332391186, 718.6751721992105, 42.43428988592416, 1291.174096268078, 33.2310023175457, 32.756437274309015, 496.8791713874787, 499.16640069393344, 499.54488141599995, 499.3745919104882, 499.7074790329489, 499.63846027667694, 499.1399779133607, 497.26094408095634, 498.3550284990341, 594.0823087429742, 596.2092119870243, 595.1897900485314, 593.7635976361436, 593.9223590124496, 744.9847464213817, 745.6392219925021, 1497.7690648708356, 1499.814231974556, 1497.576442582435, 749.6550571592171, 751.4449071387091, 753.2812481002966, 753.5503747538689, 752.4339968832023, 753.1936894328819, 752.5500841253104, 752.4021320760376, 752.6515378014217, 754.1816072940304, 748.6636404734625, 1503.0720143255267, 750.7190358205684, 750.3654284952095, 752.5388802885395, 745.9752291970018, 754.9051784227061, 744.0079701376399, 756.3681795653257, 748.2824577221021, 752.8973611183571, 749.579812973601, 748.674721246623, 754.092994832527, 750.7524154333488, 755.3645717795314, 751.6693708954643, 755.4605394273086, 752.8558734301444, 753.8584965926789, 752.9405788519473, 751.2501465351784, 754.6316575088085, 447.80032363919867, 448.14422824001946, 447.5753109726277, 446.7539387031118, 446.3860082207574, 446.6739040652723, 446.6914732344942, 459.2847313586084, 564.2024332824619, 563.623220043766, 565.6590896369455, 565.3656187683018, 565.2168869578617, 565.9114946439774, 566.9618122826348, 745.0545604620312, 744.6056691195611, 748.1810424835087, 748.6436500902537, 744.7218002622924, 746.5914919682808, 748.9090262038601, 748.0597987470957, 746.5360893881426, 748.9195735575116, 745.0806598299939, 748.3040190614566, 746.3211837150473, 749.8274731000404, 748.9109709304965, 747.394311353225, 750.5030661157139, 748.3116300925282, 753.479186921182, 748.3473349793269, 751.2391404401487, 752.2820202171046, 748.2912399989848, 751.2949527083612, 750.268170536993, 751.7802573743732, 751.9438775341657, 750.8691926952248, 750.9574402099572, 750.759104082023, 756.9831131957347, 381.68356830612987, 397.22035158230767, 397.3554129145798, 397.1662896460318, 396.6848586148739, 396.5800499931911, 397.9350866554307, 394.8286682750392, 500.271497740085, 499.3581121313433, 499.58783396306285, 499.2476886054903, 497.3317926132443, 667.5659267231679, 666.8661263466907, 666.5290303151419, 663.6250476788962, 664.8802317815057, 664.5375421685038, 664.9957707961876, 663.1611268544162, 663.0330933754019, 663.6064570447401, 661.9589132067894, 664.9783612960113, 662.8812951804991, 665.1177094125889, 661.8241076597185, 667.53019654011, 663.3572274079957, 667.4390076372945, 662.3322514827366, 666.131644190737, 1322.2635466795134, 666.284970205435, 666.6364812672375, 672.272809157146, 698.5810521066865, 700.0801975036494, 700.3345816201085, 700.1894306613157, 1404.4035428205964, 1404.1981564898695, 1403.6923539377622, 1403.8784876194425, 701.7023670786996, 745.661836494458, 743.7220136502688, 745.8791387839177, 743.1888745841474, 748.0922219218301, 746.1200771172469, 751.6709954349614, 748.416503938293, 748.1713099376294, 747.3192714123577, 751.1240934931511, 751.6579950993528, 375.52634213701015, 375.2236096336982, 374.8548979784653, 374.1874539484677, 373.9899030097271, 373.6180723879684, 374.08054781472066, 374.21213642506103, 374.10792028396196, 374.0600931395752, 373.5999950036015, 374.724236217338, 374.14508310252177, 375.2550921128146, 374.44940078418233, 372.867388215359, 374.2048840621626, 374.42427846832504, 373.98033376236697, 374.8471563073391, 373.2648218185059, 374.9428683263428, 374.4348960297981, 374.1463845478157, 375.17437865059696, 373.8523582745291, 374.6995571072142, 374.73031931970974, 374.50912217143406, 374.7815494268223, 374.19300173358556, 371.97903901228915]
#confidence = [0, 0.059360355, 0.060854882, 0.06414756, 0.046429068, 0.050670087, 0.07347357, 0.008277357, 0.050723284, 0.18685713, 0.01250869, 0.4894451, 0.05716151, 0.93844074, 0.92349, 0.92795587, 0.9260611, 0.94536024, 0.9514474, 0.9423541, 0.9585783, 0.9142637, 0.93823254, 0.9314947, 0.9290562, 0.9167699, 0.91559947, 0.17430675, 0.6893328, 0.6803393, 0.5548705, 0.56940955, 0.7842102, 0.890044, 0.881711, 0.87115717, 0.8270296, 0.8484021, 0.84403276, 0.8552903, 0.8043296, 0.86208344, 0.8124206, 0.5677651, 0.85519266, 0.6879989, 0.8041227, 0.83815074, 0.78300154, 0.8807087, 0.8680141, 0.87717736, 0.8501404, 0.9069095, 0.8459791, 0.89617974, 0.91510963, 0.8782819, 0.8991395, 0.5798409, 0.90939224, 0.820993, 0.88985693, 0.85634273, 0.8465915, 0.9377227, 0.9278798, 0.9352946, 0.9504101, 0.94775677, 0.9076074, 0.92853373, 0.737422, 0.9465524, 0.9555316, 0.9556781, 0.95817006, 0.956915, 0.94625485, 0.87313545, 0.6023638, 0.8764101, 0.84025776, 0.849162, 0.8935652, 0.5464627, 0.8361627, 0.8462899, 0.84800076, 0.5659861, 0.7556414, 0.676845, 0.6366365, 0.7186432, 0.80461603, 0.78748065, 0.8051709, 0.87722665, 0.8968415, 0.8793681, 0.8973197, 0.9042782, 0.9053509, 0.89958155, 0.90851647, 0.8904648, 0.91609615, 0.90673554, 0.9071213, 0.84425557, 0.8660093, 0.5606215, 0.9698238, 0.97230697, 0.97510123, 0.96468174, 0.96556866, 0.97101855, 0.9473443, 0.94191766, 0.9419684, 0.9497305, 0.9459114, 0.95512474, 0.8515808, 0.8571423, 0.8871205, 0.94200915, 0.9490771, 0.9543097, 0.94020665, 0.86903787, 0.95867664, 0.95888615, 0.9687495, 0.9454547, 0.9493946, 0.93333256, 0.9471835, 0.92106813, 0.95758533, 0.90474933, 0.9662136, 0.9107944, 0.7413515, 0.92957664, 0.9187567, 0.78234506, 0.8769848, 0.9110689, 0.89276165, 0.83870816, 0.6527877, 0.85761267, 0.9508972, 0.959134, 0.8417057, 0.57748324, 0.6607003, 0.8281353, 0.7206495, 0.60607696, 0.71749026, 0.710739, 0.85122365, 0.7193301, 0.8786219, 0.90078247, 0.9121635, 0.9334358, 0.9186481, 0.9176954, 0.86050534, 0.7970458, 0.73680115, 0.7488791, 0.813664, 0.80591106, 0.8404542, 0.84114766, 0.92151797, 0.9174783, 0.9501952, 0.9441166, 0.87667, 0.93755746, 0.9376906, 0.9289652, 0.9443934, 0.8551878, 0.9535259, 0.93701416, 0.9348755, 0.9508705, 0.9027388, 0.9463606, 0.9335449, 0.9292135, 0.9399862, 0.9105155, 0.7359161]

def frequency_to_note(freq):
	if(freq <= 0):
		return 0
	return(round(12*math.log(freq/440, 2)))

fqs = []
l = len(frequency)
for i in range(l):
	if(confidence[i] > 0.75):
		fqs.append(frequency_to_note(frequency[i]))

print(fqs)

scale = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

notes = []

for fq in fqs:
	notes.append(scale[fq % 12] + str(int((fq+9)/12) + 4))



def remove_duplicates(arr):
	arr2 = []
	for i in range(len(arr)-1):
		if(arr[i] != arr[i+1]):
			arr2.append(arr[i])
	if(arr2[-1] != arr[-1]):
		arr2.append(arr[-1])
	return(arr2)


notes = remove_duplicates(notes)
print(notes)