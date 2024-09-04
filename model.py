import os
from llama_index.llms.together import TogetherLLM
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline

def llm_model(content):

    llm = TogetherLLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo", 
        api_key=os.environ.get("TOGETHER_API_KEY")
    )

    prompt_str = """Please provide a concise and clear summary of the following content. If the content is in English or another language, respond in English or that language. Make sure the summary captures the main points and key meaning of the original content:

Content: {the_content}

"""
    prompt_tmpl = PromptTemplate(prompt_str)

    pipe = QueryPipeline(chain=[prompt_tmpl, llm], reversed = True)
    response = pipe.run(content)

    return response


def llm_model_vi(content):

    llm = TogetherLLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo", 
        api_key=os.environ.get("TOGETHER_API_KEY")
    )

    prompt_str = """Please provide a concise and clear summary of the following content. If the content is in Vietnamese, respond in Vietnamese. Ensure that the summary captures the main points and key meaning of the original content:

Content: {the_content}


"""
    prompt_tmpl = PromptTemplate(prompt_str)

    pipe = QueryPipeline(chain=[prompt_tmpl, llm], reversed = True)
    response = pipe.run(content)

    return response

if __name__ == "__main__":

#     content = """
#         The Overstory is a novel by Richard Powers published in 2018 by W. W. Norton & Company. It is Powers' twelfth novel. The book is about nine Americans whose unique life experiences with trees bring them together to address the destruction of forests. Powers was inspired to write the work while teaching at Stanford University, after he encountered giant redwood trees for the first time.[1]
#         The Overstory was a contender for multiple awards. It was shortlisted for the 2018 Man Booker Prize on September 20, 2018[2] and won the 2019 Pulitzer Prize for Fiction on April 15, 2019,[3] as well as the William Dean Howells Medal in 2020. Reviews of the novel have been mostly positive, with praise of the structure, writing, and compelling reading experience.[4]
#         Patricia Westerford, one of the novel's central characters, was heavily inspired by the life and work of forest ecologist Suzanne Simard.[5][6][7] Westerford pens a popular science book, The Secret Forest, whose title alludes to real-world books such as The Hidden Life of Trees: What They Feel, How They Communicate – Discoveries from a Secret World by German forester Peter Wohlleben, The Secret Life of Trees by British science writer Colin Tudge, and Finding the Mother Tree by Simard herself.[8]
#         Plot
#         The Overstory is divided into four sections, titled "roots", "trunk", "crown", and "seeds", mirroring the structure of a tree.
#         Roots
#         On the Hoel farm in Iowa, a sentinel chestnut tree survives the blight, becoming a beloved tree for four generations of Hoels. The last Hoel, Nicholas, returns to the farm on Christmas day to find his entire family has perished from propane asphyxiation.
#         Winston Ma, a Chinese American, has three daughters. To honor his own father, Winston plants a mulberry tree. When he takes his own life beneath the mulberry, his eldest daughter, Mimi, is left to divide the family heirlooms, which includes three jade rings and an ancient scroll.
#         In the Appich family, a tree is planted for each child. Adam's tree is a maple. After Adam's older sister goes missing, he loses interest in his insect avocation, and his grades slip. His passion for knowledge is rekindled after finding a book on social psychology. While there, he learns about the bystander effect.
#         Ray Brinkman and Dorothy Cazaly begin dating, but Dorothy struggles with commitment because she sees it as a form of ownership. They finally get married and make plans to plant something in their yard every year on their anniversary.
#         After enduring the Stanford prison experiment, Douglas Pavlicek enlists in the U.S. Air Force. Doug's plane is shot down, and he falls out into a banyan tree. He is discharged and becomes a caretaker on a horse ranch. While driving through Oregon, he is disturbed by the sight of clear-cut hillsides. He takes a job planting thousands of Douglas fir seedlings.
#         As a boy, Neelay Mehta becomes obsessed with computers. While climbing a California live oak, Neelay falls and becomes paralyzed from the waist down. While at Stanford University, he receives inspiration from the campus trees and decides to create an immersive world video game where players conquer, expand, and interact over game content.
#         Patricia Westerford learns about trees from her father. She studies botany and forestry in college. While completing research, she discovers trees communicate with each other through chemicals. Her findings are denounced by a few prominent scientists. She loses her job and retreats into solitary life, nearly killing herself. Later, she meets two scientists who tell her that her research has been redeemed in the scientific community. She joins them at their research station and begins investigating trees once more.
#         While at college studying Actuarial Science, Olivia Vandergriff is accidentally electrocuted and her heart stops.
#         Trunk
#         After ninety seconds, Olivia's heart restarts. She senses beings of light who want her to leave school to join activists working to save the California redwoods. On the way, she stops at Nick Hoel's farm, and he agrees to go with her.
#         Mimi and Douglas are brought together when a grove of ponderosa pines is cut down in the middle of the night. Together, they decide to join activists to defend trees, both getting arrested. Eventually, they make their way to the same activist group Olivia and Nick join.
#         Neelay starts his own company, becoming extremely successful with his Mastery games. Patricia writes a bestselling book called The Secret Forest. Adam decides he is going to write his dissertation on the psychological profiles of environmental activists. Dorothy and Ray struggle to conceive a child, and Dorothy begins having an affair.
#         Nick and Olivia take their turn living in the branches of a giant redwood called Mimas to protect it from logging. Their vigil lasts for months instead of days. Dorothy asks Ray for a divorce, but Ray suddenly has a stroke and almost dies.
#         Adam interviews Nick and Olivia atop Mimas. While there, they are threatened by a helicopter and forced down; all three are sent to jail and Mimas is cut down. Adam decides to join Nick, Olivia, Mimi, and Doug in Oregon. Each takes on a new name: Mimi is Mulberry, Doug is Doug-fir, Adam is Maple, Nick is Watchman, and Olivia is Maidenhair. Frustrated with their progress to stop old-growth logging, the group decides to take matters into their own hands and begin burning logging equipment. During their final arson attempt, the explosion goes awry, and Olivia is fatally injured.
#         Crown
#         Olivia dies from the explosion, and the rest of the group flee the scene. Adam returns to graduate school and becomes a respected professor in the field of psychology. Nick lives a transient life, making activist art. Mimi changes her name and becomes a therapist. Doug lives in the secluded remnants of a mining town in Montana.
#         Dorothy cares for Ray after his stroke. They bond over their plants, especially the trees they planted in their yard. Patricia starts a seed vault to preserve trees that will soon be extinct. Neelay becomes unhappy with his Mastery games and wants to use technology to learn how to preserve the natural world.
#         A tourist finds Doug's journal with information about the group's arson activities, and Doug is arrested. To protect Mimi, he decides to identify Adam as an accomplice.
# """

#     response = llm_model(content)
#     print(response)

#     content = """
#       Ngành giáo dục Việt Nam trong những năm qua đã có những thành tích đáng khâm phục. Tuy nhiên bên cạnh đó vẫn còn tồn tại nhiều thách thức chưa thể giải quyết triệt để như bạo lực học đường, gian lận thi cử, vô lễ với giáo viên… Trong đó vấn nạn gian lận trong thi cử được xem là đáng báo động đối với giáo dục. 
#       Gian lận trong thi cử là gì? Gian lận trong thi cử là hành vi làm trái so với quy định của học sinh như quay cóp bài, mang tài liệu vào phòng thi, chạy tiền của để đạt được điểm cao. Gian lận không chỉ diễn ra ở học sinh mà còn diễn ra ở giáo viên và phụ huynh. Chính phụ huynh, giáo viên đang “dọn đường” cho học sinh, tiếp tay để học sinh gian lận. Đây thực sự là điều rất đáng buồn. 
#       Biểu hiện của gian lận trong thi cử hiện nay không phải giấu kín mà nó hiển hiện ra rất lộ liễu, hơn hết có nhiều người biết nhưng mà cũng không lên tiếng. Gian lận trong thi cử sẽ gây ra nhiều tác hại xấu cho học sinh, làm hư học sinh, khiến các em luôn ở trong tâm thế sống ỷ lại, dựa dẫm, không có ý chí vươn lên phấn đấu giành thành tích. Bao thế hệ học sinh đi qua là bất nhiêu thế hệ còn tồn tại thói xấu gian lận đáng phải bài trừ này. Hậu quả mà việc gian lận trong thi cử gây ra rất lớn, hiện tượng này có thể phá hủy tương lai còn dài của các em. Chỉ vì các em đã quen với việc gian lận, quen với việc được nâng đỡ cũng đã khiến cho các em lười tư duy, vận động để đạt kết quả tốt. 
#       Để có thể hạn chế được hiện tượng này thì thầy cô giáo cần phải nghiêm khắc và xử lí mạnh tay hơn nữa những thành phần dám vi phạm. Có như thế thì học sinh mới có thể nghiêm túc làm bài, không dựa dẫm. Thế chủ động đó sẽ khiến cho các em có thể nắm vững được kiến thức thật chắc và thật sâu. Tình trạng gian lận ở ngành giáo dục nước ta đang còn nhiều, không chỉ kiểm tra ở trường mà còn tại các kỳ thi tốt nghiệp, thi đại học cũng không hiếm. Các em đã không thể tự khẳng định được năng lực học của mình mà chỉ lo chạy theo cái danh vọng hão huyền, không thực tế. Gian lận thi cử sẽ tạo nên bệnh thành tích cần phải bài trừ. 
# """

#     response = llm_model_vi(content)
#     print(response)
    pass