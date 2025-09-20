from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
import torch

from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging
import re
from tqdm import tqdm
import os
from pathlib import Path

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# 保持你的字符替换映射不变
char_rep_map = {
    "：": ",",
    "；": ",",
    ";": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    # "\n": " ",
    "·": "-",
    "、": ",",
    "...": "…",
    ",,,": "…",
    "，，，": "…",
    "……": "…",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
    # ":": ",",
    "〇": "零",
    "○": "零",
}

def replace_chars(full_script, char_rep_map):
    result = ''
    for char in full_script:
        result += char_rep_map.get(char, char)
    return result

def combine_to_max_length(combined_sentences: list, max_length: int = 400):
    """
    Combines a list of sentences into new strings that do not exceed a maximum length.

    Args:
        combined_sentences: A list of string sentences.
        max_length: The maximum character length for each combined string.

    Returns:
        A list of combined strings.
    """
    if not combined_sentences:
        return []

    result_list = []
    current_string = ""

    for sentence in combined_sentences:
        # **优化：检查单个句子是否已超过最大长度**
        if len(sentence) > max_length:
            logger.warning(f"Warning: A single sentence exceeds the max_length ({len(sentence)} > {max_length}). It will be added as a separate item.")
            result_list.append(sentence)
            continue # 跳过下面的逻辑，直接处理下一个句子

        # Check if adding the new sentence exceeds the max length
        # We add 1 for the space separator
        if len(current_string) + len(sentence) + 1 <= max_length:
            # If the current string is not empty, add a space
            if current_string:
                current_string += " " + sentence
            else:
                current_string = sentence
        else:
            # If it would exceed, finalize the current string and start a new one
            result_list.append(current_string)
            current_string = sentence
            
    # Add the last combined string if it's not empty
    if current_string:
        result_list.append(current_string)
        
    return result_list

def process_line(s_line: str):
    s_line = replace_chars(s_line, char_rep_map)
    # 修正：更新re.split的正则表达式，以包含所有可能的分隔符
    # 你的 char_rep_map 中把"？"和"！"转换成了英文问号和感叹号，所以保留它们
    # 同时，它也把"。"转换成了"."，所以也保留
    # 还需要添加中文的“。！？”，以防转换不完全
    sentences = re.split('([?!.？！])', s_line)
    
    # re.split 的一种更简洁的替代方案是 re.findall(r'[^?!.]+[?!.]', s_line)
    # 但你目前的代码逻辑是可行的，只是下面需要重新拼接
    
    sentences_with_punct = [s for s in sentences if s]
    temp_str = ""
    combined_sentences = []
    for s in sentences_with_punct:
        # 你的 char_rep_map 只将部分中文标点转换为英文
        # 修正：这里需要同时检查中文和英文标点
        if s in ['?', '!', '.', '？', '！', '。']: 
            temp_str += s
            combined_sentences.append(temp_str)
            temp_str = ""
        else:
            if temp_str:
                combined_sentences.append(temp_str)
            temp_str = s

    if temp_str:
        combined_sentences.append(temp_str)

    return combined_sentences

def gererator_speech(to_tts_txt, model, processor, voice_samples):
    
    for _index, _line in tqdm(enumerate(to_tts_txt)):
        output_path = f"output/1p_tiyan-2_{_index}.wav"
        output_txt_path = f"output/1p_tiyan-2_{_index}.txt"
        if os.path.exists(output_path):
            print(f'dest output file exists[{output_path}] continue to process next ...')
            continue
        output_txt_path = Path(f"output/1p_tiyan-2_{_index}.txt")
        # Create the parent directory
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to the file
        output_txt_path.write_text(_line, encoding='utf-8')

        inputs = processor(
            text=[_line],  # Wrap in list for batch processing
            voice_samples=[voice_samples],  # Wrap in list for batch processing
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.4,
            tokenizer=processor.tokenizer,
            # generation_config={'do_sample': True, 'temperature': 0.95, 'top_p': 0.95, 'top_k': 0},
            generation_config={'do_sample': False},
            verbose=True,
        )
        
        processor.save_audio(
            outputs.speech_outputs[0],  # First (and only) batch item
            output_path=output_path,
        )
        print(f'finish process ouput file : {output_path}')

def main():
    input_txt = "/Users/larry/ai/mnist/tiyan/1p_tiyan-others.txt"
    voice_samples = ["/Users/larry/coderesp/VibeVoice/demo/voices/zh-phi0_woman.WAV"]
    max_length = 400

    model_path = "VibeVoice-1.5B"
    device="mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"use device[{device}]")

    processor = VibeVoiceProcessor.from_pretrained(model_path)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation='sdpa')

    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)
    f_lines = []
    with open(input_txt, 'r') as f:
        f_lines = f.readlines()
    s_line = "".join(f_lines)
    
    to_tts_txt = combine_to_max_length(process_line(s_line), max_length=max_length)
    print(to_tts_txt)

    gererator_speech(to_tts_txt, model, processor, voice_samples)
    

def test_split():
    str_hello = """靠引用马克思和列宁的词句写成的文章到处泛滥，只能起教条解说员作用的著作屡见不鲜。这些情况说明，在他们的头脑里首先有一个教条，然后再用这个教条去衡量现实，这就是他们的思维方法。这种思维方法是由于日本人长期处在受天皇制政权镇压的历史条件下，而且对苏联社会主义及对斯大林肃反的事实一无所知而逐渐形成的。
如果在五十年代，日本人了解到斯大林晚年进行的可怕的肃反的事实真相，如果预见到六十万拘留者所吃的苦头，实际上是斯大林肃反政策的一个组成部分，那么，一九五六年从批判斯大林当中所受到的巨大冲击，以及所采取的对应措施也就自然不同了。
但是在日本根本不可能出现安德烈·纪德那样的人物。他早在三十年代就大胆地发表了《苏联游记》一书。《马克思主义和语言学问题》(一九五零年)，《苏联社会主义经济问题》(一九五二年)被宣传为斯大林晚年的天才著作。这也被我国盲目地接受了。
这种作法也是很说明问题的。马克思主义本来应该是科学，为什么始终都是非科学的呢？原因之一是由于突出了观念，同时又轻视现实。往往具体的事实只能成为装璜理想的材料，而决不会有更大的用处。因此，常常产生一种错觉，似乎观念与现实是完全一致的。
因为只注意搜集那些有助于说明观念的材料，所以这种虚构使人感到材料是符合观念的。斯大林的肃反充分采用了这种虚构的做法。流亡国外的俄国人写的报道文学和回忆录，在欧洲和美国已堆积如山，但这些著作都被当作反苏反共的宣传而无人理睬。现在我还记得一位马克思主义者在一篇文章中说，肃反这类蛊惑人心的宣传，好象早上的露水一样，在阳光下瞬息间就会消失。
对他们来说，根本不存在什么对社会主义产生怀疑的问题，在他们看来，社会主义就是善，而资本主义就是恶，二者必居其一。这种明快的逻辑，在社会主义处于坚如磐石般的团结的时代，似乎是很有说服力的，但是对斯大林展开批判之后，就一举崩溃了。
由于把马克思主义当做教条，从而给日本的革命实践带来严重损失的事例，当然不只是在“左”倾冒险主义时代出现过一次。从我肤浅的体会来说，如果当时对这个问题能够大胆地加以解剖，出现了变革的热情，那么，肯定会清醒地认识到，对教条的盲从将会带来多么悲惨的结局。
在我手头有一本蜡纸油印的《占领下日本的农村调查报告》，是一九五二年八月发行的，全书共三百零六页，是一部以农村问题研究会的名义编写的值得纪念的农村调查记录。这份报告是对山村、平原、军事基地、开垦村、渔村等十六个村庄进行的调查记录。这份记录是作者试图把毛泽东的农村解放区原原本本地搬到日本来的实践的证据。以山村工作队为代表的农民解放运动，完全照搬了毛泽东的阶级划分的标准。
然而，在土改后的现实的日本农村很难找到雇农和农村无产阶级。为在农村创建山村工作队，或者说为在农村建立工农联盟的堡垒而进行的实际调查及其他工作，其结局都是很惨的。
许多山村工作队在所要解放的山村处于孤立无援的状态。他们常常在夜间偷偷地出去开展党的组织工作。我不能否定青年们献身于山村工作队的那种热情。那是当代的民意派，他们学习当地的方言，用方言交谈，抓住谈话中的要点，帮助贫农干活，也想听取他们的要求，然后组织斗争。青年们的这种热情与历来由官方自上而下进行的调查相比较，无论在方法上，还是在精神上都有根本的区别。他们和农民同吃同住，照看小孩的学习，甚至把连环画和幻灯也带到了农村。
这种调查方法，尽管未能定型，但它是农村调查的划时代的尝试。在某种程度上是从中国革命中学来的。与一九六零年反对日美安全条约时的“还乡运动”相比较，至少在持久性和稳定性方面可以说具有历史性意义。我也作为农村调查团的一名团员去了山梨县的一个山村。那里的农民是极其贫困的，他们由于霜冻和冰雹的灾害而负了债，一天领取一百八十日元的失业救济金过着穷困的生活。
""".replace('\n', '')
    res = combine_to_max_length(process_line(str_hello), max_length=200)
    print(f'res length: {len(res)}')
    for _line in res:
        print(_line)

if __name__ == "__main__":
    main()
    # test_split()