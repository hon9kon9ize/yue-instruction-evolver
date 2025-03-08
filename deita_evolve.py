from urllib.parse import quote
import argparse
import json
import os
from datasets import load_dataset
import functools
import datetime
import glob
from tqdm.auto import tqdm
import pandas as pd
from typing import List
import requests
import random
from tenacity import retry, stop_after_attempt, wait_fixed

GOOGLE_AISTUDIO_API_KEY = "YOU_API_KEY"
MODEL_NAME = "gemini-2.0-flash"

hans_chars = set(
    "万与专业丛东丝丢两严丧个丬丰临为丽举么义乌乐乔习乡书买乱争于亏亘亚产亩亲亵亸亿仅从仑仓仪们价众优会伛伞伟传伤伥伦伧伪伫体佥侠侣侥侦侧侨侩侪侬俣俦俨俩俪俭债倾偬偻偾偿傥傧储傩儿兑兖党兰关兴兹养兽冁内冈册写军农冢冯冲决况冻净凄凉减凑凛凤凫凭凯击凼凿刍刘则刚创删别刬刭刽刿剀剂剐剑剥剧劝办务劢动励劲劳势勋勐勚匀匦匮区医华协单卖卢卤卫却卺厂厅历厉压厌厍厕厢厣厦厨厩厮县参叆叇双发变叙叠叶号叹叽吁吕吗吣吨听启吴呒呓呕呖呗员呙呛呜咏咙咛咝咤咴哌哑哒哓哔哕哗哙哜哝哟唛唝唠唡唢唣唤唿啧啬啭啮啰啴啸喷喽喾嗫嗳嘘嘤嘱噜嚣嚯团园囱围囵国图圆圣圹场坂坏块坚坛坜坝坞坟坠垄垅垆垒垦垧垩垫垭垯垱垲垴埘埙埚埝埯堑堕塆墙壮声壳壶壸处备复够头夸夹夺奁奂奋奖奥妆妇妈妩妪妫姗娄娅娆娇娈娱娲娴婳婴婵婶媪嫒嫔嫱嬷孙学孪宁宝实宠审宪宫宽宾寝对寻导寿将尔尘尧尴尸尽层屃屉届属屡屦屿岁岂岖岗岘岙岚岛岭岽岿峃峄峡峣峤峥峦崂崃崄崭嵘嵚嵛嵝嵴巅巩巯币帅师帏帐帘帜带帧帮帱帻帼幂幞并广庄庆庐庑库应庙庞废庼廪开异弃张弥弪弯弹强归当录彟彦彻径徕忆忏忧忾怀态怂怃怄怅怆怜总怼怿恋恳恶恸恹恺恻恼恽悦悫悬悭悯惊惧惨惩惫惬惭惮惯愍愠愤愦愿慑慭憷懑懒懔戆戋戏戗战戬户扦执扩扪扫扬扰抚抛抟抠抡抢护报担拟拢拣拥拦拧拨择挂挚挛挜挝挞挟挠挡挢挣挤挥挦捞损捡换捣据掳掴掷掸掺掼揽揿搀搁搂搅携摄摅摆摇摈摊撄撵撷撸撺擞攒敌敛数斋斓斩断无旧时旷旸昙昼昽显晋晒晓晔晕晖暂暧术机杀杂权条来杨杩极构枞枢枣枥枧枨枪枫枭柜柠柽栀栅标栈栉栊栋栌栎栏树栖样栾桊桠桡桢档桤桥桦桧桨桩梦梼梾检棂椁椟椠椤椭楼榄榇榈榉槚槛槟槠横樯樱橥橱橹橼檩欢欤欧歼殁殇残殒殓殚殡殴毁毂毕毙毡毵氇气氢氩氲汇汉汤汹沓沟没沣沤沥沦沧沨沩沪沵泞泪泶泷泸泺泻泼泽泾洁洒洼浃浅浆浇浈浉浊测浍济浏浐浑浒浓浔浕涛涝涞涟涠涡涢涣涤润涧涨涩淀渊渌渍渎渐渑渔渖渗温湾湿溃溅溆溇滗滚滞滟滠满滢滤滥滦滨滩滪漤潆潇潋潍潜潴澜濑濒灏灭灯灵灾灿炀炉炖炜炝点炼炽烁烂烃烛烟烦烧烨烩烫烬热焕焖焘煅煳熘爱爷牍牦牵牺犊犟状犷犸犹狈狍狝狞独狭狮狯狰狱狲猃猎猕猡猪猫猬献獭玑玙玚玛玮环现玱玺珉珏珐珑珰珲琎琏琐琼瑶瑷璎瓒瓮瓯电画畅畲畴疖疗疟疠疡疬疮疯疴痈痉痒痖痨痪痫瘅瘆瘗瘘瘪瘫瘾瘿癞癣癫癯皑皱皲盏盐监盖盗盘眍眦眬睁睐睑瞒瞩矫矶矾矿砀码砖砗砚砜砺砻砾础硁硕硖硗硙硚确硷碍碛碜碱碹磙礼祎祢祯祷祸禀禄禅离秃秆种积称秽秾稆税稣稳穑穷窃窍窑窜窝窥窦窭竖竞笃笋笔笕笺笼笾筑筚筛筜筝筹签简箓箦箧箨箩箪箫篑篓篮篱簖籁籴类籼粜粝粤粪粮糁糇紧絷纟纠纡红纣纤纥约级纨纩纪纫纬纭纮纯纰纱纲纳纴纵纶纷纸纹纺纻纼纽纾线绀绁绂练组绅细织终绉绊绋绌绍绎经绐绑绒结绔绕绖绗绘给绚绛络绝绞统绠绡绢绣绤绥绦继绨绩绪绫绬续绮绯绰绱绲绳维绵绶绷绸绹绺绻综绽绾绿缀缁缂缃缄缅缆缇缈缉缊缋缌缍缎缏缐缑缒缓缔缕编缗缘缙缚缛缜缝缞缟缠缡缢缣缤缥缦缧缨缩缪缫缬缭缮缯缰缱缲缳缴缵罂网罗罚罢罴羁羟羡翘翙翚耢耧耸耻聂聋职聍联聩聪肃肠肤肷肾肿胀胁胆胜胧胨胪胫胶脉脍脏脐脑脓脔脚脱脶脸腊腌腘腭腻腼腽腾膑臜舆舣舰舱舻艰艳艹艺节芈芗芜芦苁苇苈苋苌苍苎苏苘苹茎茏茑茔茕茧荆荐荙荚荛荜荞荟荠荡荣荤荥荦荧荨荩荪荫荬荭荮药莅莜莱莲莳莴莶获莸莹莺莼萚萝萤营萦萧萨葱蒇蒉蒋蒌蓝蓟蓠蓣蓥蓦蔷蔹蔺蔼蕲蕴薮藁藓虏虑虚虫虬虮虽虾虿蚀蚁蚂蚕蚝蚬蛊蛎蛏蛮蛰蛱蛲蛳蛴蜕蜗蜡蝇蝈蝉蝎蝼蝾螀螨蟏衅衔补衬衮袄袅袆袜袭袯装裆裈裢裣裤裥褛褴襁襕见观觃规觅视觇览觉觊觋觌觍觎觏觐觑觞触觯詟誉誊讠计订讣认讥讦讧讨让讪讫训议讯记讱讲讳讴讵讶讷许讹论讻讼讽设访诀证诂诃评诅识诇诈诉诊诋诌词诎诏诐译诒诓诔试诖诗诘诙诚诛诜话诞诟诠诡询诣诤该详诧诨诩诪诫诬语诮误诰诱诲诳说诵诶请诸诹诺读诼诽课诿谀谁谂调谄谅谆谇谈谊谋谌谍谎谏谐谑谒谓谔谕谖谗谘谙谚谛谜谝谞谟谠谡谢谣谤谥谦谧谨谩谪谫谬谭谮谯谰谱谲谳谴谵谶豮贝贞负贠贡财责贤败账货质贩贪贫贬购贮贯贰贱贲贳贴贵贶贷贸费贺贻贼贽贾贿赀赁赂赃资赅赆赇赈赉赊赋赌赍赎赏赐赑赒赓赔赕赖赗赘赙赚赛赜赝赞赟赠赡赢赣赪赵赶趋趱趸跃跄跖跞践跶跷跸跹跻踊踌踪踬踯蹑蹒蹰蹿躏躜躯车轧轨轩轪轫转轭轮软轰轱轲轳轴轵轶轷轸轹轺轻轼载轾轿辀辁辂较辄辅辆辇辈辉辊辋辌辍辎辏辐辑辒输辔辕辖辗辘辙辚辞辩辫边辽达迁过迈运还这进远违连迟迩迳迹适选逊递逦逻遗遥邓邝邬邮邹邺邻郄郏郐郑郓郦郧郸酝酦酱酽酾酿释鉴銮錾钆钇针钉钊钋钌钍钎钏钐钑钒钓钔钕钖钗钘钙钚钛钝钞钟钠钡钢钣钤钥钦钧钨钩钪钫钬钭钮钯钰钱钲钳钴钵钶钷钸钹钺钻钼钽钾钿铀铁铂铃铄铅铆铈铉铊铋铍铎铏铐铑铒铕铗铘铙铚铛铜铝铞铟铠铡铢铣铤铥铦铧铨铪铫铬铭铮铯铰铱铲铳铴铵银铷铸铹铺铻铼铽链铿销锁锂锃锄锅锆锇锈锉锊锋锌锍锎锏锐锑锒锓锔锕锖锗错锚锜锞锟锠锡锢锣锤锥锦锨锩锫锬锭键锯锰锱锲锳锴锵锶锷锸锹锺锻锼锽锾锿镀镁镂镃镆镇镈镉镊镌镍镎镏镐镑镒镕镖镗镙镚镛镜镝镞镟镠镡镢镣镤镥镦镧镨镩镪镫镬镭镮镯镰镱镲镳镴镶长门闩闪闫闬闭问闯闰闱闲闳间闵闶闷闸闹闺闻闼闽闾闿阀阁阂阃阄阅阆阇阈阉阊阋阌阍阎阏阐阑阒阓阔阕阖阗阘阙阚阛队阳阴阵阶际陆陇陈陉陕陧陨险随隐隶隽难雏雠雳雾霁霭靓静靥鞑鞒鞯鞴韦韧韨韩韪韫韬韵页顶顷顸项顺须顼顽顾顿颀颁颂颃预颅领颇颈颉颊颋颌颍颎颏颐频颒颓颔颕颖颗题颙颚颛颜额颞颟颠颡颢颣颤颥颦颧风飏飐飑飒飓飔飕飖飗飘飙飚飞飨餍饤饥饦饧饨饩饪饫饬饭饮饯饰饱饲饳饴饵饶饷饸饹饺饻饼饽饾饿馀馁馂馃馄馅馆馇馈馉馊馋馌馍馎馏馐馑馒馓馔馕马驭驮驯驰驱驲驳驴驵驶驷驸驹驺驻驼驽驾驿骀骁骂骃骄骅骆骇骈骉骊骋验骍骎骏骐骑骒骓骔骕骖骗骘骙骚骛骜骝骞骟骠骡骢骣骤骥骦骧髅髋髌鬓魇魉鱼鱽鱾鱿鲀鲁鲂鲄鲅鲆鲇鲈鲉鲊鲋鲌鲍鲎鲏鲐鲑鲒鲓鲔鲕鲖鲗鲘鲙鲚鲛鲜鲝鲞鲟鲠鲡鲢鲣鲤鲥鲦鲧鲨鲩鲪鲫鲬鲭鲮鲯鲰鲱鲲鲳鲴鲵鲶鲷鲸鲹鲺鲻鲼鲽鲾鲿鳀鳁鳂鳃鳄鳅鳆鳇鳈鳉鳊鳋鳌鳍鳎鳏鳐鳑鳒鳓鳔鳕鳖鳗鳘鳙鳛鳜鳝鳞鳟鳠鳡鳢鳣鸟鸠鸡鸢鸣鸤鸥鸦鸧鸨鸩鸪鸫鸬鸭鸮鸯鸰鸱鸲鸳鸴鸵鸶鸷鸸鸹鸺鸻鸼鸽鸾鸿鹀鹁鹂鹃鹄鹅鹆鹇鹈鹉鹊鹋鹌鹍鹎鹏鹐鹑鹒鹓鹔鹕鹖鹗鹘鹚鹛鹜鹝鹞鹟鹠鹡鹢鹣鹤鹥鹦鹧鹨鹩鹪鹫鹬鹭鹯鹰鹱鹲鹳鹴鹾麦麸黄黉黡黩黪黾鼋鼌鼍鼗鼹齄齐齑齿龀龁龂龃龄龅龆龇龈龉龊龋龌龙龚龛龟咨尝"
)


def count_hans_chars(text):
    return len([c for c in text if c in hans_chars])


def get_written_lang_ratio(txt: str) -> float:
    return (txt.count("是") + txt.count("的") + txt.count("在")) / (
        txt.count("係") + txt.count("嘅") + txt.count("喺") + 1
    )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def text_generation(
    prompt: str,
    stopSequences=[],
    temperature=0.9,
    top_k=40,
    top_p=1,
    maxOutputTokens=8192,
) -> str:
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GOOGLE_AISTUDIO_API_KEY}"

    res = requests.post(
        api_url,
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "topK": top_k,
                "topP": top_p,
                "maxOutputTokens": maxOutputTokens,
                "stopSequences": stopSequences,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        },
        timeout=120,
    )

    return res.json()


EVALUATE_RESPONSE_TEMPLATE = """We would like to request your feedback on the performance of AI assistant in response to the given question displayed following.

## Tips:
Please rate according to the accuracy of the response to the instruction and the input.
Each assistant receives a score on a scale of 0 to 5, where a higher score indicates higher level of the accuracy.
You must just give a score without any other reasons.

## Question:

{prompt}

## Response:

{response}

## Score: """


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def evaluate_response(prompt: str, response: str) -> int:
    prompt = EVALUATE_RESPONSE_TEMPLATE.format(prompt=prompt, response=response)
    generated_res = text_generation(
        prompt,
        temperature=0.2,
        top_k=20,
        top_p=0.9,
        maxOutputTokens=5,
        stopSequences=["\n", "#", "*"],
    )

    if "candidates" not in generated_res or len(generated_res["candidates"]) == 0:
        print("[evaluate_response()]: no candidates")
        raise Exception("No candidates")

    if generated_res["candidates"][0]["finishReason"] != "STOP":
        print("[evaluate_response()]: not stopped")
        raise Exception("Not stopped")

    score = generated_res["candidates"][0]["content"]["parts"][0]["text"].strip()

    if len(score) == 0:
        print("[evaluate_response()]: no score")
        raise Exception("No score")

    if not score:
        print("[evaluate_response()]: no score")
        raise Exception("No score")

    # check if score is a valid number
    if not score.isdigit():
        print("[evaluate_response()]: score not a number")
        raise Exception("Score not a number")

    score = int(score)

    if score < 0 or score > 5:
        return 0

    return score


# Difficulty judge
DIFFICULTY_JUDGE_PROMPT_TEMPLATE = """We would like you to evaluate and rate the difficulty and complexity of the following question.
You should give an overall score on a scale of 1 to 5, where a higher score indicates higher difficulty and complexity.
You must just give a score without any other reasons.

## Question:

{prompt}

## Score:"""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def difficulty_judge(
    prompt: str,
) -> List[int]:
    prompt = DIFFICULTY_JUDGE_PROMPT_TEMPLATE.format(prompt=prompt)
    generated_res = text_generation(
        prompt,
        temperature=0.2,
        top_k=20,
        top_p=0.9,
        maxOutputTokens=5,
        stopSequences=["\n", "#", "*"],
    )

    if "candidates" not in generated_res or len(generated_res["candidates"]) == 0:
        print("[difficulty_judge()]: no candidates")
        raise Exception("No candidates")

    if generated_res["candidates"][0]["finishReason"] != "STOP":
        print("[difficulty_judge()]: not stopped")
        raise Exception("Not stopped")

    score = generated_res["candidates"][0]["content"]["parts"][0]["text"].strip()

    if len(score) == 0:
        print("[difficulty_judge()]: no score", score)
        raise Exception("No score")

    if not score:
        print("[difficulty_judge()]: no score", score)
        raise Exception("No score")

    # check if score is a valid number
    if not score.isdigit():
        print("[difficulty_judge()]: score not a number", score)
        raise Exception("Score not a number")

    score = int(score)

    if score < 1 or score > 5:
        return 0

    return score


# In-Breath Generation
IN_BREATH_EVOLVING_TEMPLATE = """我想你扮演一個提示創作人（Prompt Creator）。
你嘅目標係從 #Given Prompt# 中汲取靈感，去創造一個全新嘅提示。
呢個新提示應該同 #Given Prompt# 屬於同一範疇，但係要更加罕見。
#Created Prompt# 嘅長度同難度級別應該同 #Given Prompt# 相似。
#Created Prompt# 必須係合理嘅，而且必須畀人類理解同回應。
你應該用廣東話去寫 #Created Prompt#
#Given Prompt# 、#Created Prompt#、畀咗嘅提示同創造咗嘅提示唔可以喺 #Created Prompt# 中出現。

#Given Prompt#：
{prompt}

#Created Prompt#：
"""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def evolving(prompt: str):
    prompt = IN_BREATH_EVOLVING_TEMPLATE.format(prompt=prompt)
    generated_res = text_generation(
        prompt, temperature=0.7, top_k=20, top_p=0.9, maxOutputTokens=2048
    )

    if "candidates" not in generated_res or len(generated_res["candidates"]) == 0:
        print("[evolving()]: no candidates")
        raise Exception("No candidates")

    if generated_res["candidates"][0]["finishReason"] != "STOP":
        print("[evolving()]: not stopped")
        raise Exception("Not stopped")

    created_prompt = (
        generated_res["candidates"][0]["content"]["parts"][0]["text"]
        .strip()
        .replace("#Created Prompt#：", "")
        .replace("#Given Prompt#：", "")
    )

    if created_prompt == prompt:
        raise Exception("Prompt not created")

    if len(created_prompt) > 2048:
        print("[evolving()]: prompt too long")
        raise Exception("Prompt too long")

    is_zh_hans = get_written_lang_ratio(created_prompt) > 1
    has_hans_chars = count_hans_chars(created_prompt) > 1

    if is_zh_hans or has_hans_chars:
        print("[evolving()]: prompt not Cantonese")
        raise Exception("Prompt not Cantonese")

    if len(created_prompt) < 10:
        print("[evolving()]: prompt too short")
        raise Exception("Prompt too short")

    return created_prompt


# Diversity Generation
PROMPT_DEEPENING_TEMPLATE = """我想你扮演一個提示重寫器（Prompt Rewriter）。
你嘅目標係將一個畀你嘅提示改寫成一個更複雜嘅版本，令到啲出名嘅 AI 系統（例如 ChatGPT 同 GPT4）難啲處理。
但係，改寫咗嘅提示必須要合理，而且人類必須可以理解同回應。
你嘅改寫唔可以省略啲非文字部分，例如 #Given Prompt# 入面嘅表格同代碼。
另外，請唔好省略 #Given Prompt# 入面嘅輸入。
你應該用廣東話去寫 #Rewritten Prompt#
你應該用以下方法嚟複雜化畀你嘅提示：如果 #Given Prompt# 包含對某啲問題嘅查詢，可以增加查詢嘅深度同廣度。
你應該盡量唔好令到 #Rewritten Prompt# 變得冗長，#Rewritten Prompt# 只可以喺 #Given Prompt# 入面添加 10 至 20 個字。
#Given Prompt#、#Rewritten Prompt#、畀你嘅提示同重寫嘅提示唔可以喺 #Rewritten Prompt# 入面出現。

#Given Prompt#：
{prompt}

#Rewritten Prompt#：
"""

PROMPT_CONCRETIZING_TEMPLATE = """我想你扮演一個提示重寫器（Prompt Rewriter）。
你嘅目標係將一個畀你嘅提示改寫成一個更複雜嘅版本，令到啲出名嘅 AI 系統
（例如 ChatGPT 同 GPT4）難啲處理。
但係個改寫咗嘅提示一定要合理，而且人類必須理解同回應到。
你嘅改寫唔可以省略非文本部分，例如 #Given Prompt# 入面嘅表格同代碼。另外，請
唔好省略 #Given Prompt# 入面嘅輸入。
你應該用廣東話去寫 #Rewritten Prompt#
你應該用以下方法複雜化畀你嘅提示：
請用更具體嘅概念取代一般概念。或者
你應該盡量唔好令到 #Rewritten Prompt# 變得冗長，#Rewritten Prompt# 只可以喺 #Given Prompt# 入面添加 10 至 20 個字。
#Given Prompt#、#Rewritten Prompt#、畀你嘅提示同重寫嘅提示唔可以喺 #Rewritten Prompt# 入面出現。

#Given Prompt#：
{prompt}

#Rewritten Prompt#：
"""

INCREASED_REASONING_STEPS_PROMPT = """我想你扮演一個提示重寫器（Prompt Rewriter）。
你嘅目標係將一個畀你嘅提示改寫成一個更複雜嘅版本，令到啲出名嘅 AI 系統（例如 ChatGPT 同 GPT4）難啲處理。
但係，改寫咗嘅提示必須要合理，而且人類必須可以理解同回應。
你嘅改寫唔可以省略啲非文字部分，例如 #Given Prompt# 入面嘅表格同代碼。
另外，請唔好省略 #Given Prompt# 入面嘅輸入。
你**應該**用廣東話去寫 #Rewritten Prompt#
你**應該**用以下方法嚟複雜化畀你嘅提示：
如果 #Given Prompt# 只需要幾個簡單嘅思考過程就可以解決，你可以將佢改寫成明確要求多步推理。
#Given Prompt#、#Rewritten Prompt#、畀你嘅提示同重寫嘅提示唔可以喺 #Rewritten Prompt# 入面出現。

#Given Prompt#：
{prompt}

#Rewritten Prompt#：
"""

ADDING_CONSTRAINTS_PROMPT = """我想你扮演一個提示重寫器（Prompt Rewriter）。
你嘅目標係將一個畀你嘅提示改寫成一個更複雜嘅版本，令到啲出名嘅 AI 系統（例如 ChatGPT 同 GPT4）難啲處理。
但係，改寫咗嘅提示必須要合理，而且人類必須可以理解同回應。
你嘅改寫唔可以省略啲非文字部分，例如 #Given Prompt# 入面嘅表格同代碼。
另外，請唔好省略 #Given Prompt# 入面嘅輸入。
你**應該**用廣東話去寫 #Rewritten Prompt#
你**應該**用以下方法嚟複雜化畀你嘅提示：
請喺 #Given Prompt# 入面再加多一個限制/要求。
你**應該**盡量唔好令到 #Rewritten Prompt# 變長，
#Rewritten Prompt# 最多只可以喺 #Given Prompt# 入面加 10 至 20 個字。

#Given Prompt#：
{prompt}

#Rewritten Prompt#：
"""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def diverse(prompt: str):
    template = random.choice(
        [
            PROMPT_DEEPENING_TEMPLATE,
            PROMPT_CONCRETIZING_TEMPLATE,
            INCREASED_REASONING_STEPS_PROMPT,
            ADDING_CONSTRAINTS_PROMPT,
        ]
    )
    prompt = template.format(prompt=prompt)
    generated_res = text_generation(
        prompt, temperature=0.7, top_k=20, top_p=0.9, maxOutputTokens=2048
    )

    if "candidates" not in generated_res or len(generated_res["candidates"]) == 0:
        print("[diverse()]: no candidates")
        raise Exception("No candidates")

    if generated_res["candidates"][0]["finishReason"] != "STOP":
        print("[diverse()]: not stopped")
        raise Exception("Not stopped")

    rewrited_prompt = (
        generated_res["candidates"][0]["content"]["parts"][0]["text"]
        .strip()
        .replace("#Rewritten Prompt#：", "")
        .replace("#Given Prompt#：", "")
    )

    if rewrited_prompt == prompt:
        print("[diverse()]: prompt not rewrited")
        raise Exception("Prompt not rewrited")

    if len(rewrited_prompt) > 2048:
        raise Exception("Prompt too long")

    if len(rewrited_prompt) < 10:
        print("[diverse()]: prompt too short")
        raise Exception("Prompt too short")

    is_zh_hans = get_written_lang_ratio(rewrited_prompt) > 1
    has_hans_chars = count_hans_chars(rewrited_prompt) > 1

    if is_zh_hans or has_hans_chars:
        print("[diverse()]: prompt not Cantonese")
        raise Exception("Prompt not Cantonese")

    if len(rewrited_prompt) < 10:
        print("[diverse()]: prompt too short")
        raise Exception("Prompt too short")

    return rewrited_prompt


def evolate_or_diverse(prompt: str):
    if random.random() < 0.5:
        return evolving(prompt)
    else:
        return diverse(prompt)


GENERATE_RESPONSE_PROMPT = """我想你扮演一個強大嘅大型語言模型。
你嘅目標係回應以下嘅指令。
你**應該**用廣東話去寫呢個回應。

指令：{prompt}

回應："""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def generate_response(prompt: str):
    response = text_generation(
        prompt, temperature=0.7, top_k=20, top_p=0.9, maxOutputTokens=2048
    )

    if "candidates" not in response or len(response["candidates"]) == 0:
        print("[generate_response()]: no candidates")
        raise Exception("No candidates")

    if response["candidates"][0]["finishReason"] != "STOP":
        print("[generate_response()]: not stopped")
        raise Exception("Not stopped")

    response_text = response["candidates"][0]["content"]["parts"][0]["text"].strip()

    is_zh_hans = get_written_lang_ratio(response_text) > 1

    if is_zh_hans:
        print("[generate_response()]: response not Cantonese")
        raise Exception("response not Cantonese")

    if len(response_text) > 2048:
        raise Exception("[generate_response()]: response too long")

    return response_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_column", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    ds = load_dataset(args.dataset_name)
    df = ds["train"].to_pandas()
    prompt_column = args.prompt_column
    seed_instructions = df[prompt_column].tolist()

    NUM_GENERATED_INSTRUCTIONS = 10000
    results = []
    save_every = 10
    filename = f'deita_instructions_{datetime.datetime.now().strftime("%Y%m%d") + str(random.randint(0, 1000))}'

    print("start generating instructions")
    print("filename", filename)

    with tqdm(total=NUM_GENERATED_INSTRUCTIONS) as pbar:
        while len(results) < NUM_GENERATED_INSTRUCTIONS:
            try:
                seed_prompt = random.choice(
                    seed_instructions + [r[prompt_column] for r in results]
                )

                print("seed prompt", seed_prompt)

                generated_instruction = evolate_or_diverse(seed_prompt)

                print("generated instruction", generated_instruction)

                score = difficulty_judge(generated_instruction)

                print("score", score)

                if score < 3:
                    continue

                response = generate_response(generated_instruction)

                print("response", response)

                score = evaluate_response(generated_instruction, response)

                print("score", score)

                if score < 3:
                    continue

                results.append(
                    {
                        prompt_column: generated_instruction,
                        "output": response,
                    }
                )

                if len(results) % save_every == 0:
                    with open(f"outputs/{filename}.jsonl", "w") as f:
                        for result in results:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")

                pbar.update(1)

            except Exception as e:
                print(e)
