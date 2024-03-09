import json
from flask import Flask, request, jsonify
from se import search_engine

# from matplotlib import pyplot as plt
#
# first_run = [('genetics', 2.518773078918457, 0.092), ('Who is considered the "Father of the United States"?', 70, 0), ('economic', 13.240584135055542, 0.089), ('When was the United Nations founded?', 66.14292097091675, 0.092), ('video gaming', 20.76684594154358, 0.0), ('3D printing technology', 10.478022813796997, 0.0), ('Who is the author of "1984"?', 20.62777590751648, 0.0), ('bioinformatics', 1.682426929473877, 0.22), ('Who is known for proposing the heliocentric model of the solar system?', 49.144999742507935, 0.0), ('Describe the process of water erosion.', 33.59993600845337, 0.046), ('When was the Berlin Wall constructed?', 18.7369647026062, 0.0), ('What is the meaning of the term "Habeas Corpus"?', 23.34450912475586, 0.09), ('telecommunications', 2.4513027667999268, 0.106), ('internet', 3.6825432777404785, 0.098), ('What are the characteristics of a chemical element?', 12.70386290550232, 0.0), ('Describe the structure of a plant cell.', 30.520446300506592, 0.0), ('Who painted "Starry Night"?', 14.096388816833496, 0.054), ('computer', 5.975239992141724, 0.083), ("What is the structure of the Earth's layers?", 12.617396116256714, 0.0), ('When did World War II end?', 55.920636892318726, 0.08), ('When was the Gutenberg printing press invented?', 15.032341003417969, 0.0), ('medicine', 3.3362009525299072, 0.094), ('Describe the water cycle.', 23.112583875656128, 0.0), ('artificial intelligence', 6.162524938583374, 0.223), ('physics', 5.794700860977173, 0.083), ('nanotechnology', 1.6450960636138916, 0.35), ('When did the Black Death pandemic occur?', 40.19928193092346, 0.083), ('neuroscience', 1.8876228332519531, 0.092), ('snowboard', 1.8164541721343994, 0.215), ('Who is the founder of modern psychology?', 19.369436979293823, 0.0)]
#
# # our_weights = [6, 8, 8, 5, 7.5, 6.5] second round
# sec_run = [('genetics', 1.1275720596313477, 0.0), ('Who is considered the "Father of the United States"?', 48.64432692527771, 0.0), ('economic', 4.476658821105957, 0.0), ('When was the United Nations founded?', 56.09131717681885, 0.092), ('video gaming', 18.94423198699951, 0.0), ('3D printing technology', 8.699299812316895, 0.0), ('Who is the author of "1984"?', 14.955295085906982, 0.0), ('bioinformatics', 1.6386048793792725, 0.18), ('Who is known for proposing the heliocentric model of the solar system?', 42.70048809051514, 0.0), ('Describe the process of water erosion.', 27.733078002929688, 0.0), ('When was the Berlin Wall constructed?', 14.737531900405884, 0.0), ('What is the meaning of the term "Habeas Corpus"?', 19.684261083602905, 0.081), ('telecommunications', 2.109592914581299, 0.0), ('internet', 3.3724637031555176, 0.0), ('What are the characteristics of a chemical element?', 9.523159980773926, 0.0), ('Describe the structure of a plant cell.', 26.469409942626953, 0.0), ('Who painted "Starry Night"?', 11.471189022064209, 0.097), ('computer', 4.90512490272522, 0.0), ("What is the structure of the Earth's layers?", 7.678825855255127, 0.0), ('When did World War II end?', 46.182076930999756, 0.0), ('When was the Gutenberg printing press invented?', 15.053246021270752, 0.0), ('medicine', 2.766587972640991, 0.0), ('Describe the water cycle.', 14.84362006187439, 0.0), ('artificial intelligence', 7.7839438915252686, 0.141), ('physics', 4.298908710479736, 0.0), ('nanotechnology', 1.75203275680542, 0.224), ('When did the Black Death pandemic occur?', 31.5399272441864, 0.083), ('neuroscience', 1.9585261344909668, 0.083), ('snowboard', 1.8209950923919678, 0.0), ('Who is the founder of modern psychology?', 14.980406999588013, 0.0)]
#
# # our_weights = [6, 3, 8, 2, 6, 4] third run (change title to 9 when only 1 word in title)
# third_run = [('genetics', 2.5803310871124268, 0.0), ('Who is considered the "Father of the United States"?', 69.98238325119019, 0.0), ('economic', 3.6828160285949707, 0.0), ('When was the United Nations founded?', 52.054474115371704, 0.0), ('video gaming', 17.008179187774658, 0.0), ('3D printing technology', 8.80738115310669, 0.0), ('Who is the author of "1984"?', 17.942993879318237, 0.0), ('bioinformatics', 1.789529800415039, 0.0), ('Who is known for proposing the heliocentric model of the solar system?', 40.04637694358826, 0.0), ('Describe the process of water erosion.', 27.03498601913452, 0.0), ('When was the Berlin Wall constructed?', 15.526756048202515, 0.0), ('What is the meaning of the term "Habeas Corpus"?', 20.92017388343811, 0.0), ('telecommunications', 2.047179698944092, 0.0), ('internet', 2.666435956954956, 0.0), ('What are the characteristics of a chemical element?', 9.731714010238647, 0.0), ('Describe the structure of a plant cell.', 23.852478981018066, 0.0), ('Who painted "Starry Night"?', 12.490579843521118, 0.0), ('computer', 3.768558979034424, 0.0), ("What is the structure of the Earth's layers?", 10.299040079116821, 0.0), ('When did World War II end?', 44.40365481376648, 0.0), ('When was the Gutenberg printing press invented?', 14.54220700263977, 0.0), ('medicine', 3.4796547889709473, 0.0), ('Describe the water cycle.', 16.80163288116455, 0.0), ('artificial intelligence', 5.4580910205841064, 0.163), ('physics', 5.154909133911133, 0.0), ('nanotechnology', 1.984670877456665, 0.196), ('When did the Black Death pandemic occur?', 31.33183264732361, 0.0), ('neuroscience', 2.0387837886810303, 0.0), ('snowboard', 1.9473822116851807, 0.0), ('Who is the founder of modern psychology?', 15.713716983795166, 0.0)]
#
# # our_weights = [7, 1, 3, 7, 8, 7] third run (change title to 9 when only 1 word in title)
# four_run = [('genetics', 1.1900219917297363, 0.092), ('Who is considered the "Father of the United States"?', 33.24420952796936, 0.0), ('economic', 2.9996321201324463, 0.0), ('When was the United Nations founded?', 27.142395734786987, 0.0), ('video gaming', 8.884800434112549, 0.0), ('3D printing technology', 4.156853437423706, 0.0), ('Who is the author of "1984"?', 6.679639101028442, 0.0), ('bioinformatics', 0.6997957229614258, 0.333), ('Who is known for proposing the heliocentric model of the solar system?', 19.1176860332489, 0.0), ('Describe the process of water erosion.', 11.856934547424316, 0.046), ('When was the Berlin Wall constructed?', 7.215265989303589, 0.0), ('What is the meaning of the term "Habeas Corpus"?', 7.739425420761108, 0.044), ('telecommunications', 0.8553774356842041, 0.106), ('internet', 1.3262302875518799, 0.0), ('What are the characteristics of a chemical element?', 4.191669225692749, 0.0), ('Describe the structure of a plant cell.', 13.183106422424316, 0.0), ('Who painted "Starry Night"?', 5.201200723648071, 0.0), ('computer', 1.8698933124542236, 0.0), ("What is the structure of the Earth's layers?", 4.523378372192383, 0.0), ('When did World War II end?', 21.323047161102295, 0.0), ('When was the Gutenberg printing press invented?', 6.469037294387817, 0.0), ('medicine', 1.3387548923492432, 0.0), ('Describe the water cycle.', 9.750577449798584, 0.0), ('artificial intelligence', 2.418050527572632, 0.291), ('physics', 2.0633864402770996, 0.0), ('nanotechnology', 0.7197034358978271, 0.63), ('When did the Black Death pandemic occur?', 13.985163450241089, 0.046), ('neuroscience', 0.7401602268218994, 0.35), ('snowboard', 0.6842584609985352, 0.27), ('Who is the founder of modern psychology?', 6.532940864562988, 0.0)]
#
# # our_weights = [7, 1, 4, 2, 7, 8, 7] third run (change title to 9 when only 1 word in title) add binary
# five_run = [('genetics', 1.1960742473602295, 0.0), ('Who is considered the "Father of the United States"?', 35.63291597366333, 0.0), ('economic', 2.201484203338623, 0.0), ('When was the United Nations founded?', 29.594274282455444, 0.0), ('video gaming', 7.99501633644104, 0.0), ('3D printing technology', 4.173543930053711, 0.0), ('Who is the author of "1984"?', 7.7879931926727295, 0.0), ('bioinformatics', 0.8134682178497314, 0.383), ('Who is known for proposing the heliocentric model of the solar system?', 18.496349573135376, 0.0), ('Describe the process of water erosion.', 12.574583530426025, 0.0), ('When was the Berlin Wall constructed?', 6.444722890853882, 0.273), ('What is the meaning of the term "Habeas Corpus"?', 8.730129957199097, 0.268), ('telecommunications', 0.9571413993835449, 0.139), ('internet', 1.6375000476837158, 0.0), ('What are the characteristics of a chemical element?', 4.464034080505371, 0.16), ('Describe the structure of a plant cell.', 12.46362042427063, 0.081), ('Who painted "Starry Night"?', 6.2302350997924805, 0.177), ('computer', 1.874253273010254, 0.0), ("What is the structure of the Earth's layers?", 4.815925121307373, 0.0), ('When did World War II end?', 23.479413747787476, 0.0), ('When was the Gutenberg printing press invented?', 5.748034477233887, 0.0), ('medicine', 1.3995091915130615, 0.0), ('Describe the water cycle.', 10.03682279586792, 0.138), ('artificial intelligence', 2.386993408203125, 0.402), ('physics', 2.1450958251953125, 0.0), ('nanotechnology', 0.7810156345367432, 0.509), ('When did the Black Death pandemic occur?', 14.803896427154541, 0.251), ('neuroscience', 0.7525932788848877, 0.309), ('snowboard', 0.7157242298126221, 0.215), ('Who is the founder of modern psychology?', 6.01617693901062, 0.0)]
#
# # our_weights = [0, 0, 0, 0, 0, 8, 3, 4, 3, 3]
# six_run = [('genetics', 0.8698318004608154, 0.132), ('Who is considered the "Father of the United States"?', 20.510140895843506, 0.0), ('economic', 1.959218978881836, 0.044), ('When was the United Nations founded?', 14.252920389175415, 0.0), ('video gaming', 3.061460256576538, 0.0), ('3D printing technology', 2.8101134300231934, 0.095), ('Who is the author of "1984"?', 4.34236216545105, 0.0), ('bioinformatics', 0.9690885543823242, 0.383), ('Who is known for proposing the heliocentric model of the solar system?', 15.860202550888062, 0.0), ('Describe the process of water erosion.', 5.400469541549683, 0.0), ('When was the Berlin Wall constructed?', 3.4408609867095947, 0.0), ('What is the meaning of the term "Habeas Corpus"?', 4.1610212326049805, 0.0), ('telecommunications', 0.7949986457824707, 0.0), ('internet', 2.174541711807251, 0.0), ('What are the characteristics of a chemical element?', 2.357560157775879, 0.0), ('Describe the structure of a plant cell.', 4.650835990905762, 0.0), ('Who painted "Starry Night"?', 3.693203926086426, 0.177), ('computer', 1.5665805339813232, 0.0), ("What is the structure of the Earth's layers?", 3.6009721755981445, 0.0), ('When did World War II end?', 16.754169464111328, 0.0), ('When was the Gutenberg printing press invented?', 4.135575532913208, 0.0), ('medicine', 1.2488470077514648, 0.0), ('Describe the water cycle.', 4.0268166065216064, 0.0), ('artificial intelligence', 1.7635407447814941, 0.0), ('physics', 2.007932186126709, 0.0), ('nanotechnology', 0.6683194637298584, 0.509), ('When did the Black Death pandemic occur?', 8.180373430252075, 0.0), ('neuroscience', 0.7346370220184326, 0.363), ('snowboard', 0.6358785629272461, 0.209), ('Who is the founder of modern psychology?', 5.257742404937744, 0.0)]
#
# # our_weights = [0,0,0,0,0,8,3,4,1,1] change title  (no binary) to 8 when only 1 word in title and reduce body to 5
# seven_run = [('genetics', 0.7524139881134033, 0.132), ('Who is considered the "Father of the United States"?', 25.124649047851562, 0.0), ('economic', 2.2618319988250732, 0.044), ('When was the United Nations founded?', 19.11579990386963, 0.0), ('video gaming', 3.988826036453247, 0.0), ('3D printing technology', 3.5962460041046143, 0.095), ('Who is the author of "1984"?', 6.887830972671509, 0.0), ('bioinformatics', 0.6140673160552979, 0.383), ('Who is known for proposing the heliocentric model of the solar system?', 20.22899079322815, 0.0), ('Describe the process of water erosion.', 7.442148685455322, 0.0), ('When was the Berlin Wall constructed?', 5.645848989486694, 0.0), ('What is the meaning of the term "Habeas Corpus"?', 5.395991802215576, 0.0), ('telecommunications', 0.8076941967010498, 0.0), ('internet', 1.6349990367889404, 0.117), ('What are the characteristics of a chemical element?', 2.9746530055999756, 0.044), ('Describe the structure of a plant cell.', 5.860370874404907, 0.0), ('Who painted "Starry Night"?', 5.509042024612427, 0.177), ('computer', 1.8777573108673096, 0.0), ("What is the structure of the Earth's layers?", 3.398837089538574, 0.0), ('When did World War II end?', 23.196917057037354, 0.0), ('When was the Gutenberg printing press invented?', 5.371915102005005, 0.0), ('medicine', 2.4030990600585938, 0.084), ('Describe the water cycle.', 5.30096697807312, 0.0), ('artificial intelligence', 1.8742649555206299, 0.325), ('physics', 1.1154489517211914, 0.0), ('nanotechnology', 0.6634957790374756, 0.526), ('When did the Black Death pandemic occur?', 12.275413990020752, 0.0), ('neuroscience', 0.6525509357452393, 0.309), ('snowboard', 0.6604928970336914, 0.282), ('Who is the founder of modern psychology?', 6.035895109176636, 0.0)]
#
# eighth_run = [('best marvel movie', 10.357719898223877, 0.0), ('How do kids come to world?', 15.994806051254272, 0.0), ('Information retrieval', 4.067456960678101, 0.64), ('LinkedIn', 0.4893150329589844, 0.0), ('How to make coffee?', 0.9448628425598145, 0.0), ('Ritalin', 0.46217775344848633, 0.319), ('How to make wine at home?', 8.796100854873657, 0.0), ('Most expensive city in the world', 22.19297504425049, 0.0), ('India', 3.4411280155181885, 0.081), ('how to make money fast?', 3.9191250801086426, 0.054), ('Netflix', 0.648554801940918, 0.109), ('Apple computer', 2.6399247646331787, 0.0), ('The Simpsons', 0.57318115234375, 0.268), ('World cup', 15.859612941741943, 0.0), ('How to lose weight?', 2.2416880130767822, 0.0), ('Java', 0.8322641849517822, 0.0), ('Air Jordan', 4.623019695281982, 0.116), ('how to deal with depression?', 3.9922831058502197, 0.0), ('How do you make gold', 3.3157761096954346, 0.0), ('Marijuana', 0.626885175704956, 0.112), ('How to make hummus', 0.47068285942077637, 0.194), ('Winter', 2.325824022293091, 0.0), ('Rick and Morty', 1.1944389343261719, 0.0), ('Natural Language processing', 6.407603025436401, 0.0), ('World Cup 2022', 16.640356302261353, 0.0), ('Dolly the sheep', 1.33024001121521, 0.139), ('What is the best place to live in?', 19.76019287109375, 0.0), ('Elon musk', 1.0008738040924072, 0.194), ('How do you breed flowers?', 1.632390022277832, 0.0)]
#
# runs = [first_run, sec_run, third_run, four_run, five_run, six_run, seven_run, eighth_run]
# avg_results = []
# avg_times = []
# for run in runs:
#     avg_res = sum(map(lambda x: x[2], run)) / len(run)
#     avg_time = sum(map(lambda x: x[1], run)) / len(run)
#     avg_results.append(avg_res)
#     avg_times.append(avg_time)
#
#
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(avg_results) + 1), avg_results, marker='o', color='blue')
# plt.title('Average Results per Run')
# plt.xlabel('Run')
# plt.ylabel('Average Result')
# plt.grid(True)
# plt.show()
#
# # Plotting average times
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(avg_times) + 1), avg_times, marker='x', color='red')
# plt.title('Average Times per Run')
# plt.xlabel('Run')
# plt.ylabel('Average Time')
# plt.grid(True)
# plt.show()

runs = [([0.98, 0.84, 0.66, 0.92, 0.7000000000000001, 0.42, 0.9400000000000001, 0.1, 0.8, 0.98], 13.689796415964762, 0.11946666666666667),
([0.04, 0.64, 0.28, 0.9, 0.96, 0.96, 0.28, 0.76, 0.74, 1.0], 11.088580419277323, 0.14282758620689656),
([0.46, 0.78, 0.54, 0.86, 0.2, 0.9400000000000001, 0.6, 0.02, 0.54, 0.34], 11.137397609908005, 0.07348275862068965),
([0.12, 0.74, 0.42, 0.86, 0.68, 0.4, 0.2, 0.44, 0.22, 0.64], 11.087290854289614, 0.14882758620689654),
([0.18, 0.12, 0.66, 0.28, 0.2, 0.8200000000000001, 0.54, 0.42, 0.9, 0.9], 11.013641168331278, 0.062344827586206894),
([0.24, 0.16, 0.78, 0.84, 0.9, 0.96, 0.68, 0.26, 0.12, 0.14], 10.990611133904293, 0.10051724137931034),
([0.6, 0.5, 0.06, 0.8200000000000001, 0.4, 0.36, 0.58, 0.48, 0.56, 0.02], 11.159021311792834, 0.15934482758620688),
([0.84, 0.52, 0.04, 0.3, 0.38, 0.68, 0.62, 0.68, 0.16, 0.7000000000000001], 10.977481307654545, 0.13786206896551725),
([0.92, 0.62, 0.14, 0.52, 0.8, 0.02, 0.6, 0.8, 0.46, 0.14], 10.938921681765851, 0.1798620689655172),
([0.66, 0.74, 0.08, 1.0, 0.22, 0.36, 0.52, 0.66, 0.84, 1.0], 10.82236622941905, 0.1626896551724138),
([0.8200000000000001, 0.88, 0.66, 0.64, 0.12, 0.16, 0.22, 0.38, 0.8200000000000001, 0.58], 10.960776723664383, 0.11124137931034482),
([0.18, 0.62, 0.18, 0.9400000000000001, 0.7000000000000001, 0.52, 0.54, 0.88, 0.68, 0.58], 10.970272557488803, 0.16882758620689658),
([0.74, 0.6, 0.68, 0.52, 0.14, 0.52, 0.92, 0.84, 1.0, 0.3], 11.109629795469086, 0.12358620689655173),
([0.96, 0.34, 0.46, 0.96, 0.92, 0.8200000000000001, 0.62, 0.64, 1.0, 0.6], 11.186221459816242, 0.14420689655172414),
([0.9400000000000001, 0.76, 0.74, 0.6, 0.92, 0.08, 0.64, 1.0, 0.9400000000000001, 0.32], 11.439883963815097, 0.16117241379310346),
([0.3, 0.62, 0.86, 0.14, 0.64, 0.76, 0.48, 0.8, 0.72, 0.52], 11.198279298585037, 0.09424137931034482),
([0.96, 0.36, 0.48, 0.92, 0.9, 0.24, 0.64, 0.84, 0.74, 0.9], 11.149403892714401, 0.16758620689655174),
([0.1, 0.16, 1.0, 0.7000000000000001, 0.78, 0.24, 0.18, 0.62, 0.7000000000000001, 0.36], 11.275624176551556, 0.12575862068965515),
([0.26, 0.52, 0.38, 0.04, 0.68, 0.54, 0.62, 0.44, 0.04, 0.92], 11.080236788453727, 0.13606896551724137),
([0.76, 0.64, 0.54, 0.4, 0.64, 0.04, 0.78, 1.0, 0.88, 0.28], 11.043659136213105, 0.17113793103448277),
([0.14, 0.72, 0.3, 0.7000000000000001, 1.0, 0.5, 0.52, 0.92, 0.96, 0.76], 11.084112035817114, 0.17244827586206898),
([0.34, 0.56, 0.86, 0.14, 0.28, 0.78, 0.44, 0.18, 0.54, 0.36], 11.104571745313446, 0.037137931034482756),
([0.68, 0.02, 0.64, 0.22, 0.66, 0.1, 0.8200000000000001, 0.56, 0.46, 0.12], 11.040221929550171, 0.13896551724137932),
([0.76, 0.7000000000000001, 0.28, 0.74, 0.88, 0.36, 0.42, 0.36, 0.84, 0.88], 11.25316125771095, 0.16644827586206892),
([0.14, 0.72, 0.52, 0.12, 0.12, 0.06, 0.12, 0.98, 0.52, 0.28], 11.295091694798963, 0.11137931034482759),
([0.16, 0.14, 0.8, 0.62, 0.68, 0.14, 0.32, 0.9, 0.56, 0.44], 11.338583074767014, 0.13813793103448277),
([0.08, 0.66, 0.16, 0.62, 1.0, 0.04, 0.7000000000000001, 0.9400000000000001, 0.02, 0.64], 11.269601698579459, 0.17889655172413793),
([0.98, 0.38, 0.34, 0.84, 1.0, 0.34, 0.58, 0.04, 0.24, 0.32], 11.509211490894186, 0.1446206896551724),
([0.52, 0.98, 0.16, 0.04, 0.5, 0.52, 0.22, 0.6, 0.1, 0.14], 11.301023771022928, 0.10351724137931036),
([0.32, 0.5, 0.3, 0.46, 0.4, 0.74, 0.92, 0.64, 0.32, 0.08], 10.976505772820834, 0.1306551724137931),
([0.16, 0.64, 0.88, 0.86, 0.6, 0.1, 0.38, 0.84, 0.5, 0.64], 11.071737478519308, 0.14872413793103448),
([0.3, 0.58, 0.62, 0.26, 0.42, 0.16, 0.24, 0.18, 0.22, 0.12], 11.16361562959079, 0.11003448275862067),
([0.54, 0.18, 0.18, 0.2, 0.48, 0.8200000000000001, 0.52, 0.6, 0.26, 0.4], 11.005539417266846, 0.1163103448275862),
([0.02, 0.5, 0.06, 0.28, 0.06, 0.5, 0.5, 0.28, 0.56, 0.46], 11.06977474278417, 0.11472413793103448),
([0.24, 0.44, 0.12, 0.48, 0.72, 0.2, 0.9400000000000001, 0.98, 0.88, 0.44], 11.278734092054696, 0.18434482758620688),
([0.64, 0.46, 0.02, 0.8, 0.66, 0.62, 0.4, 0.9, 0.66, 0.5], 11.062514074917498, 0.15817241379310348),
([0.54, 0.06, 0.48, 0.32, 0.1, 0.98, 0.3, 0.66, 0.8, 0.28], 11.193524574411326, 0.053310344827586204),
([0.74, 1.0, 0.52, 0.88, 0.06, 0.74, 0.38, 0.42, 0.02, 0.2], 10.00573991084921, 0.06786206896551725),
([0.52, 0.5, 0.28, 0.02, 0.28, 0.04, 0.76, 0.28, 0.02, 0.24], 11.648084517182975, 0.15224137931034482),
([0.92, 0.46, 0.34, 0.36, 1.0, 0.44, 0.8200000000000001, 0.66, 0.84, 0.5], 11.57455561900961, 0.15651724137931036),
([0.14, 0.8200000000000001, 0.44, 0.88, 0.9, 0.48, 0.3, 0.86, 0.64, 0.8], 11.570015019383924, 0.16406896551724137),
([0.14, 0.62, 1.0, 0.16, 0.64, 0.28, 0.44, 0.28, 0.38, 0.08], 11.786538132305804, 0.09148275862068964),
([0.76, 0.38, 0.44, 0.86, 0.48, 0.72, 0.2, 0.44, 0.74, 0.48], 11.166166864592453, 0.11427586206896553),
([0.06, 0.44, 0.3, 0.84, 0.42, 0.16, 0.58, 0.58, 0.44, 0.04], 11.110133162860212, 0.15755172413793103),
([0.8200000000000001, 0.46, 0.3, 0.28, 0.48, 0.8, 0.78, 0.28, 0.54, 0.66], 11.044554200665704, 0.10520689655172415),
([0.8200000000000001, 0.18, 0.32, 0.02, 0.5, 0.7000000000000001, 0.74, 0.78, 0.8, 0.44], 11.805763548818128, 0.12379310344827586),
([0.8, 0.38, 0.58, 0.16, 0.92, 0.2, 0.42, 0.28, 0.56, 0.06], 12.212332240466413, 0.13382758620689655),
([0.3, 0.84, 0.6, 0.8, 0.88, 0.12, 1.0, 0.04, 0.28, 0.54], 11.883862240561124, 0.1460689655172414),
([0.14, 0.4, 0.1, 0.42, 0.58, 0.12, 1.0, 0.62, 0.34, 0.92], 10.854427666499697, 0.18268965517241378),
([0.32, 0.1, 0.14, 0.5, 0.96, 0.88, 0.8200000000000001, 0.78, 0.68, 0.44], 11.168963711837243, 0.14093103448275865),
([0.98, 0.66, 0.92, 0.8200000000000001, 0.38, 0.12, 0.36, 0.34, 0.64, 0.78], 10.91803998782717, 0.11572413793103448),
([0.6, 0.1, 0.8200000000000001, 0.36, 0.24, 0.22, 0.8200000000000001, 0.96, 0.8200000000000001, 0.26], 10.903475991610822, 0.12779310344827588),
([0.9, 0.92, 0.76, 0.42, 0.08, 0.44, 0.2, 0.7000000000000001, 0.64, 1.0], 10.859384544964495, 0.08817241379310346),
([0.02, 0.64, 0.34, 0.9, 0.8200000000000001, 0.68, 0.98, 0.28, 0.52, 0.08], 10.852596874894767, 0.14324137931034486),
([0.42, 0.3, 0.72, 0.14, 0.18, 0.42, 0.56, 0.9400000000000001, 0.06, 0.76], 10.378285720430572, 0.10589655172413794)]
scores = ['body_bm25_bi', 'title_bm25_bi' ,
                      'body_bm25_stem' , 'title_bm25_stem' ,
                      'title_binary_stem' , 'body_bm25_no_stem' ,
                      'title_bm25_no_stem' ,
                      'title_binary_no_stem', 'pr', 'pv' ]
# print(scores)
# fsdf = list(sorted(runs, key=lambda x: x[2], reverse=True))[:10]
# for tup in fsdf:
#     print(tup[0], tup[2])

#{'body_bm25_bi': 0.98, 'title_bm25_bi': 0.84, 'body_bm25_stem': 0.66, 'title_bm25_stem': 0.92, 'title_binary_stem': 0.7000000000000001, 'body_bm25_no_stem': 0.42, 'title_bm25_no_stem': 0.9400000000000001, 'title_binary_no_stem': 0.1, 'pr': 0.8, 'pv': 0.98}