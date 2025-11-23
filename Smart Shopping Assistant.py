from typing import TypedDict, Annotated, List, Literal, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from camoufox.sync_api import Camoufox
from langchain_community.tools.youtube.search import YouTubeSearchTool
from youtubesearchpython import VideosSearch




load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


llm = ChatGroq(model = "llama-3.3-70b-versatile")


query_input = input("\nYour Message: ")
prompt = f"لطفا از متن زیر فقط نام محصول را استخراج کن.\nمتن: {query_input}"
q_resp = llm.invoke(prompt).content


class SupervisorState(MessagesState):
    """State for the multi-agent"""
    next_agent: str = "ناظر"
    torob_data: str = ""
    basalam_data: str = ""
    snap_data: str = ""
    technolife_data: str = ""
    digikala_data: str = ""
    youtube_data: str = ""
    final_report: str = ""
    task_complete: bool = False
    current_task: str = ""


def create_supervisor_chain():
    """Creates the supervisor decision chain"""
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """تو در نقش یک ناظر هستی که یک تیم از ایجنت ها را مدیریت می کند:

1. ترب: محصولات موجود در سایت ترب را پیدا می کند
2. باسلام: محصولات موجود در سایت باسلام را پیدا می کند
3. اسنپ: محصولات موجود در سایت اسنپ را پیدا می کند
4. تکنولایف: محصولات موجود در سایت تکنولایف را پیدا می کند
5. دیجیکالا: محصولات موجود در سایت دیجیکالا را پیدا می کند
6. یوتیوب: لینک های مربوط به موضوع را پیدا می کند
7. نویسنده: گزارشی از متن ارائه می کند

بر اساس حالت فعلی و گفتگو، تصمیم بگیر کدام ایجنت باید در حالت بعدی مورد استفاده قرار گیرد.
اگر وظیفه انجام شد، با کلمه "انجام شد" پاسخ بده.

حالت فعلی:
- آیا داده ترب دارد: {has_torob}
- آیا داده باسلام دارد: {has_basalam}
- آیا داده اسنپ دارد: {has_snap}
- آیا داده تکنولایف دارد: {has_technolife}
- آیا داده دیجیکالا دارد: {has_digikala}
- آیا داده یوتیوب دارد: {has_youtube}
- آیا داده گزارش دارد: {has_report}

فقط نام ایجنت، یعنی "ترب" یا "باسلام" یا "اسنپ" یا "تکنولایف" یا "دیجیکالا" یا "نویسنده" یا "یوتیوب" یا "انجام شد" را در پاسخ ارائه کن.
"""),
        ("human", "{task}")
    ])

    return supervisor_prompt | llm




def supervisor_agent(state: SupervisorState) -> Dict:
    messages = state.get("messages", [])
    current_task = state.get("current_task", "")

    if not current_task:
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        if human_msgs:
            current_task = human_msgs[-1].content.strip()
        else:
            current_task = "هیچ وظیفه ای وجود ندارد"

    has_torob = bool(state.get("torob_data"))
    has_basalam = bool(state.get("basalam_data"))
    has_snap = bool(state.get("snap_data"))
    has_technolife = bool(state.get("technolife_data"))
    has_digikala = bool(state.get("digikala_data"))
    has_youtube = bool(state.get("youtube_data"))
    has_report = bool(state.get("final_report"))

    chain = create_supervisor_chain()
    decision = chain.invoke({
        "task": current_task,
        "has_torob": has_torob,
        "has_basalam": has_basalam,
        "has_snap": has_snap,
        "has_technolife": has_technolife,
        "has_digikala": has_digikala,
        "has_youtube": has_youtube,
        "has_report": has_report
    })

    decision_text = decision.content.strip().lower()
    print(f"Supervisor decision: {decision_text}")

    if "انجام شد" in decision_text or has_report:
        next_agent = "__end__"
        supervisor_msg = "ناظر: تمام وظایف کامل شد! کار تیمی عالی بود."
    elif "باسلام" in decision_text or (has_torob and not has_basalam):
        next_agent = "باسلام"
        supervisor_msg = "ناظر: داده های ترب جمع آوری شد. اکنون زمان جمع آوری داده ها از باسلام است."
    elif "اسنپ" in decision_text or (has_basalam and not has_snap):
        next_agent = "اسنپ"
        supervisor_msg = "ناظر: داده های باسلام جمع آوری شد. اکنون زمان جمع آوری داده ها از اسنپ است."
    elif "تکنولایف" in decision_text or (has_snap and not has_technolife):
        next_agent = "تکنولایف"
        supervisor_msg = "ناظر: داده های اسنپ جمع آوری شد. اکنون زمان جمع آوری داده ها از تکنولایف است."
    elif "دیجیکالا" in decision_text or (has_technolife and not has_digikala):
        next_agent = "دیجیکالا"
        supervisor_msg = "ناظر: داده های تکنولایف جمع آوری شد. اکنون زمان جمع آوری داده ها از دیجیکالا است."
    elif "یوتیوب" in decision_text or (has_digikala and not has_youtube):
        next_agent = "یوتیوب"
        supervisor_msg = "ناظر: داده های دیجیکالا جمع آوری شد. اکنون زمان جمع آوری داده ها از یوتیوب است."
    elif "نویسنده" in decision_text or (has_youtube and not has_report):
        next_agent = "نویسنده"
        supervisor_msg = "ناظر: جمع آوری داده ها تمام شد. در حال نگارش گزارش نهایی."
    else:
        next_agent = "ترب"
        supervisor_msg = "ناظر: شروع با ترب برای جمع‌آوری اطلاعات."

    return {
        "messages": state["messages"] + [AIMessage(content=supervisor_msg)],
        "next_agent": next_agent,
        "current_task": current_task  
    }



def torob_agent(state: SupervisorState) -> Dict:
    """Researcher uses Groq to gather information"""
    
    inf = []
    prices = []

    with Camoufox() as browser:
        page = browser.new_page()
        page.goto('https://torob.com/')
        
        page.wait_for_selector('input#search-query-input', timeout=10000)
        page.click('input#search-query-input')
        page.keyboard.type(q_resp)
        page.wait_for_timeout(1000)
        page.keyboard.press('Enter')

        page.wait_for_timeout(5000)

        for _ in range(3): 
            page.evaluate("window.scrollBy(0, window.innerHeight);")
            page.wait_for_timeout(1200)  

        for i in page.query_selector_all('h2.ProductCard_desktop_product-name__JwqeK'):
            inf.append(i.inner_text())

        for i in page.query_selector_all('div.ProductCard_desktop_product-price-text__y20OV'):
            prices.append(i.inner_text())
        
    li = list(zip(inf, prices))
    lines = [f"{inf} | {prices}" for inf, prices in li]
    f =  "\n".join(lines)
    
    agent_message = f"""ترب: من داده ها را از سایت ترب جمع آوری کردم.\n\nداده های جمع آوری شده:\n{f}"""

    return {
        "messages": state["messages"] + [AIMessage(content = agent_message)],
        "torob_data": f,
        "next_agent": "ناظر"
    }




def basalam_agent(state: SupervisorState) -> Dict:
    """Summarizer uses Groq to summarize the research"""
    inf = []

    with Camoufox() as browser:
        page = browser.new_page()
        page.goto('https://basalam.com/')
        page.wait_for_timeout(5000)
        page.click('input[name="q"]')
        page.keyboard.type(q_resp)
        page.wait_for_timeout(3000)
        page.keyboard.press('Enter')
        page.wait_for_timeout(5000)
        for i in page.query_selector_all('div.ds6KtE'):
            inf.append(i.inner_text())

    prompt = f"از متن زیر فقط نام محصولات را به همراه قیمتشان در یک فرمت مناسب به من بده.\n متن: {inf}"
    resp = llm.invoke(prompt)
    response = resp.content
    
    agent_message = f"باسلام: من داده ها را سایت باسلام جمع آوری کردم.\n\nداده های جمع آوری شده:\n{response}"

    return {
        "messages": state["messages"] + [AIMessage(content = agent_message)],
        "basalam_data": response,
        "next_agent": "ناظر"
    }




def snap_agent(state: SupervisorState) -> Dict:
    inf = []
    prices = []

    with Camoufox() as browser:
        page = browser.new_page()
        page.goto('https://snappshop.ir/')
        
        page.wait_for_selector('input#ss-search-input', timeout=10000)
        page.click('input#ss-search-input')
        page.keyboard.type(q_resp)
        page.wait_for_timeout(1000)
        page.keyboard.press('Enter')

        page.wait_for_timeout(5000)

        for _ in range(3): 
            page.evaluate("window.scrollBy(0, window.innerHeight);")
            page.wait_for_timeout(1200)  

        for i in page.query_selector_all(
            'h3.ProductCard_product-card__title__nZguw.line-clamp-2.text-gray-700.overflow-hidden.mb-xs'
        ):
            inf.append(i.inner_text())

        page.wait_for_selector('p.price span.text-bold', timeout=10000)
        for i in page.query_selector_all('p.price span.text-bold'):
            prices.append(i.inner_text())
        
    li = list(zip(inf, prices))
    lines = [f"{inf} | {prices}" for inf, prices in li]
    f =  "\n".join(lines)
    
    agent_message = f"اسنپ: من داده ها را از سایت اسنپ جمع آوری کردم.\n\nداده های جمع آوری شده:\n{f}"

    return {
        "messages": state["messages"] + [AIMessage(content = agent_message)],
        "snap_data": f,
        "next_agent": "ناظر"
    }




def technolife_agent(state: SupervisorState) -> Dict:
    inf = []
    prices = []

    with Camoufox() as browser:
        page = browser.new_page()
        page.goto('https://www.technolife.com/')
        page.wait_for_selector('input[placeholder*="جستجو"]', timeout=10000)
        
        page.click('input[placeholder*="جستجو"]')
        page.keyboard.type(q_resp)
        page.keyboard.press('Enter')

        page.wait_for_selector('h2.yekanbakh-en', timeout=20000)
        page.wait_for_timeout(3000)

        for _ in range(3): 
            page.evaluate("window.scrollBy(0, window.innerHeight);")
            page.wait_for_timeout(1500)  

        for i in page.query_selector_all('h2.yekanbakh-en'):
            inf.append(i.inner_text())
        
        for i in page.query_selector_all('p.text-\\[22px\\].font-semiBold.leading-5.text-primary-shade-1'):
            prices.append(i.inner_text())
        
    li = list(zip(inf, prices))
    lines = [f"{inf} | {prices}" for inf, prices in li]
    f =  "\n".join(lines)
    
    agent_message = f"تکنولایف: من داده ها را از سایت تکنولایف جمع آوری کردم.\n\nداده های جمع آوری شده:\n{f}"

    return {
        "messages": state["messages"] + [AIMessage(content = agent_message)],
        "technolife_data": f,
        "next_agent": "ناظر"
    }




def digikala_agent(state: SupervisorState) -> Dict:
    inf = []
    prices = []

    with Camoufox() as browser:
        page = browser.new_page()
        page.goto('https://www.digikala.com/')

        page.click('span[data-cro-id="searchbox-type"]')
        page.wait_for_timeout(1000)

        page.wait_for_selector('input[name="search-input"]', timeout=10000)
        page.click('input[name="search-input"]')
        page.keyboard.type(q_resp)
        page.keyboard.press('Enter')

        page.wait_for_timeout(6000)

        for _ in range(3):  
            page.evaluate("window.scrollBy(0, window.innerHeight);")
            page.wait_for_timeout(1200)  

        for i in page.query_selector_all(
            'h3.ellipsis-2.text-body2-strong.text-neutral-700.styles_VerticalProductCard__productTitle__6zjjN'
        ):
            inf.append(i.inner_text())

        for i in page.query_selector_all('span[data-testid="price-final"]'):
            prices.append(i.inner_text())

    li = list(zip(inf, prices))
    lines = [f"{inf} | {prices}" for inf, prices in li]
    f =  "\n".join(lines)
    
    agent_message = f"دیجیکالا: من داده ها را از سایت دیجیکالا جمع آوری کردم.\n\nداده های جمع آوری شده:\n{f}"

    return {
        "messages": state["messages"] + [AIMessage(content = agent_message)],
        "digikala_data": f,
        "next_agent": "ناظر"
    }




def youtube_agent(state: SupervisorState) -> Dict:
    """youtube agent provide youtube links"""

    s = []
    videosSearch = VideosSearch(q_resp, limit=2)
    results = videosSearch.result()

    for video in results['result']:
        s.append(video['link'])

    agent_message = f"یوتیوب: من لینک های ویدیوی مربوط به محصول را جمع آوری کردم.\n\nلینک های جمع آوری شده:\n{s}"
    
    return {
        "messages": state["messages"] + [AIMessage(content = agent_message)],
        "youtube_data": s,
        "next_agent": "ناظر"
    }




def writer_agent(state: SupervisorState) -> Dict:
    """Writer uses Groq to create final report"""
    torob_data = state.get("torob_data", "")
    basalam_data = state.get("basalam_data", "")
    snap_data = state.get("snap_data", "")
    technolife_data = state.get("technolife_data", "")
    digikala_data = state.get("digikala_data", "")
    youtube_data = state.get("youtube_data")
    task = state.get("current_task")

    writing_prompt = f"""
تو یک دستیار خبره در حوزه بازاریابی و پیشنهاد دهنده خرید هستی. از هر سایت فروش اینترنتی کالا، محصولاتی همراه با قیمتشان به تو ارائه می شود.
فقط محصولاتی را در نظر بگیر که نام آن در متن وظیفه آمده است. 

در بخش مقدمه، یک مقدمه جامع و دقیق در مورد محصولی که نام آن در متن وظیفه آمده است بنویس. هدف از این بخش، آشنایی کاربر با محصول مورد نظر است. این مقدمه باید جنبه های فنی محصول مورد نظر را نیز در بر گیرد. بخش مقدمه باید 500 کلمه باشد.

در بخش بعدی، یعنی بخش تحلیل و بررسی قیمت ها، باید فقط محصولاتی از هر سایت که نام آن در متن وظیفه آمده است را در نظر بگیری و بر اساس شباهت ها، ویژگی های کیفی، عوامل کیفی، عملکرد، تاثیر، نکات کلیدی و سایر عوامل هر محصول را همراه باسایت ارائه دهنده آن محصول را در یک جدول خلاصه کنی و یک توضیح جامع و دقیق در مورد جدول ارائه شده و اینکه چه مفهومی بیان می دارد، ارائه دهی. دقت داشته باش باید ویژگی های فنی هر محصول را نیز در جدول قید کنی.

در بخش پیشنهادات، باید تمام محصولاتی که هم نام با محصولی که کاربر در متن وظیفه به آن محصول اشاره کرده است را در نظر بگیری و با توجه به ویژگی های فنی محصول، آن محصولی که کمترین قیمت را نسبت به محصولات مشابه دیگر دارد را پیدا کنی و به همراه سایت آن ارائه دهی و به کاربر توضیح دهی چرا باید آن محصول را خریداری کند تا کمترین قیمت را بپردازد. دقت داشته باش که باید ویژکی های فنی محصول انتخاب شده را نیز توضیح دهی. به علاوه، از بین تمامی محصولات موجود هم نام با محصولی که در متن وظیفه است، آن محصولی که از نظر مشخصات فنی از سایر محصولات هم نام با محصولی که در متن وظیفه است، بهتر و با کیفیت تر است را نیز به کاربر پیشنهاد کن و کاربر را قانع کن که اگر می خواهد یک محصول باکیفیت تر خریداری کند، می تواند آن محصول را نیز در نظر بگیرد. دقت داشته باش همراه با محصولی که معرفی می کنی، قیمت آرا بگویی و همچنین بگویی از کدام سایت آن را بخرد. در ادامه این بخش، اگر کاربر در متن وظیفه، مقدار بودجه خودش را برای خرید محصول مورد نظر را معین کرده بود، سه مورد از بهترین محصولات از نظر مشخصات فنی که متناسب با بودجه کاربر است را نیز به کاربر پیشنهاد بده و بگو که از چه سایتی می تواند آن ها را بخرد. طول این بخش باید 1000 کلمه باشد.

در انتها نیز لینک ویدیوی ارائه شده به تو را به کاربر ارائه کن و توضیح بده که باید برای آشنایی با محصول مورد نظر، آن را تماشا کند.  

وظیفه: {task}

داده ترب: {torob_data}

داده باسلام: {basalam_data}

داده اسنپ: {snap_data}

داده تکنولایف: {technolife_data}

داده های دیجیکالا: {digikala_data}

داده یوتیوب: {youtube_data}

برای تفکیک بخش های مختلف گزارش از جدا کننده "---" استفاده کنید. این جدا کننده ها باید در ابتدا و انتهای هر بخش قرار گیرد.
"""
    
    report_response = llm.invoke([HumanMessage(content = writing_prompt)])
    report = report_response.content

    final_report = f"""
    گزارش نهایی
{'=' * 50}
موضوع: {task}
{'=' * 50}

{report}
"""
    
    return {
        "messages": state["messages"] + [AIMessage(content = "نویسنده: گزارش کامل شد! سند کامل در زیر قابل دیدن است")],
        "final_report": final_report,
        "next_agent": "ناظر",
        "task_complete": True
    }



def router(state: SupervisorState):
    next_agent = state.get("next_agent", "ناظر")
    
    if next_agent in ["__end__", "end"] or state.get("task_complete", False):
        return END
    
    if next_agent in ["ناظر","یوتیوب" ,"تکنولایف" ,"دیجیکالا" ,"اسنپ", "ترب", "باسلام", "نویسنده"]:
        return next_agent
    
    return "ناظر"




workflow = StateGraph(SupervisorState)

workflow.add_node("ناظر", supervisor_agent)
workflow.add_node("ترب", torob_agent)
workflow.add_node("باسلام", basalam_agent)
workflow.add_node("اسنپ", snap_agent)
workflow.add_node("تکنولایف", technolife_agent)
workflow.add_node("دیجیکالا", digikala_agent)
workflow.add_node("نویسنده", writer_agent)
workflow.add_node("یوتیوب", youtube_agent)

workflow.set_entry_point("ناظر")

for node in ["ناظر","یوتیوب" ,"دیجیکالا" ,"تکنولایف","اسنپ","ترب", "باسلام", "نویسنده"]:
    workflow.add_conditional_edges(
        node,
        router,
        {
            "ناظر": "ناظر",
            "ترب": "ترب",
            "باسلام": "باسلام",
            "اسنپ": "اسنپ",
            "تکنولایف": "تکنولایف",
            "دیجیکالا": "دیجیکالا",
            "یوتیوب": "یوتیوب",
            "نویسنده": "نویسنده",
            "__end__": END
        }
    )

graph = workflow.compile()



response = graph.invoke({
    "messages": [HumanMessage(content = query_input)]
})



torob_output = response['messages'][2].content
basalam_output = response['messages'][4].content
writer_output = response['messages'][6].content
snap_output = response['messages'][8].content
technolife_output = response['messages'][10].content
digikala_output = response['messages'][12].content
youtube_output = response['messages'][14].content
final_report = response['final_report']

print("Torob Output:\n", torob_output)
print("Basalam Output:\n", basalam_output)
print("Snap Output:\n", snap_output)
print("Technolife Output:\n", technolife_output)
print("Digikala Output:\n", digikala_output)
print("Writer Output:\n", writer_output)
print("Youtube output:\n", youtube_output)
print(final_report)