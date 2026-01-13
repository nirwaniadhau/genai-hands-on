# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv

# load_dotenv()

# model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

# result =model.invoke("What is the capital of India?")
# print(result.content)


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict, Annotated,Optional,Literal

model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

class Review(TypedDict):

    key_themes: Annotated[str,"Write down the key themes discussed in the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment:Annotated[Literal["pos","neg"],"Return sentiment of the review either negative, positive or neutral"]
    pros:Annotated[Optional[list[str]],"List the pros mentioned in the review"]
    cons:Annotated[Optional[list[str]],"List the cons mentioned in the review"]


structured_model=model.with_structured_output(Review)



result= structured_model.invoke("""I’ve been using this phone for a little over three months now, and while the performance has generally been solid, the experience hasn’t been as flawless as I initially expected. The display is bright and sharp, especially indoors, but in direct sunlight I sometimes struggle with visibility, even at full brightness. The battery life is decent — on most days I get through a full day with moderate usage — but if I do gaming or heavy multitasking, the battery drains noticeably faster and I often find myself charging it in the evening.

The camera is a mixed bag for me. In good lighting, the photos come out really crisp and detailed, but in low-light situations the images tend to look a bit soft and sometimes the colors appear slightly washed out. The ultra-wide lens is useful, but there is a bit of distortion at the edges. The selfie camera is fine for casual use, though not impressive.

One thing I really like is the overall smoothness of the UI — app switching is fast, animations feel fluid, and I haven’t faced any major crashes. However, after one of the recent updates, I noticed occasional frame drops when scrolling in some apps, which can be a little annoying. The phone also tends to heat up slightly during longer gaming sessions, nothing extreme, but definitely noticeable.

Build quality feels premium and the device sits comfortably in the hand, although it is a bit on the heavier side. The speakers are loud but sound slightly tinny at higher volumes. Network performance has been reliable so far, and 5G speeds are quite good in my area.

Overall, it’s a capable device that does most things well, but it’s not perfect — there are a few rough edges that you really start noticing only after using it for a while.""")

# print(result)
# print(result['summary'])
print(result['sentiment'])