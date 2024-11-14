import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retrieval import case_search

import streamlit as st

load_dotenv()

llm = ChatOpenAI(
  model="gpt-4o", 
  temperature=0, 
  api_key=os.getenv('OPENAI_API_KEY'),
  max_tokens=4096
)

def make_document(
  system00: str,
  system01: str,
  system02: str,
  system03: str,
  system04: str,
  system05: str,
  system06: str,
  system07: str,
  system08: str,
  system10: str,
  system11: str,
  system12: str,
  system13: str,
  system14: str,
  system15: str,
  system16: str,
  system17: str,
  system18: str,
  system19: str,
  system20: str
):
  system_template = f"""
    You act as the following prompts:

    # Legal Notice Drafting Team - Agentic Workflow Prompt

    You are part of a specialized team tasked with drafting legal notices. The team consists of:

    1. Legal Expert (Sam)
    2. Document Drafter (Jenny)
    3. Format Reviewer (Tom)
    4. Final Reviewer (Team Leader) (Lisa)

    ## Team Objective

    Create accurate and effective legal notices based on user-provided information.

    ## Process

    1. **Information Gathering and Analysis** (Lisa)
      - Collect the following from the user: 
        0. Q. 안녕하세요? 이름을 알려주세요.
           A. {system00}
        1. Q. 어떤 사기 피해를 입으셨나요?
           A. {system01}
        2. Q. 피고소인의 직업이나 신분에 대해 알려주세요.
           A. {system02}
        3. Q. 고객님의 신분도 알려주세요.
           A. {system03}
        4. Q. 사건이 일어난 일시는 언제인가요? 
           A. {system04}
        5. Q. 사건이 일어난 장소는 어디인가요?
           A. {system05}
        6. Q. 고객께서는 판매자입니까 구매자입니까?
           A. {system06}
        7. Q. 처음에 피고소인(들)이 어떻게 접근했습니까? (피고소인을 알게 된 경위)
           A. {system07}
        8. Q. 어떤 중고거래 플랫폼을 사용했습니까?
           A. {system08}
        9. Q. 피고소인들은 실명을 인증하고 휴대폰 번호가 등록된 정식 회원이었습니까?
           A. {system09}
        10. Q. 피고소인이 무엇을 판매한다고 하던가요?(구매자인 경우) 또는 고객께서는 무엇을 판매하려고 어떤 글을 올렸습니까?(판매자인 경우)
            A. {system10}
        11. Q. 어떤 방식으로 상품을 받고 돈을 지불하는 걸로 결정했었나요?
            A. {system11}
        12. Q. 피고소인 뭐라고 거짓말을 하여 고소인을 속이던가요?
            A. {system12}
        13. Q. 재산은 어떻게 마련했습니까?(구매자인 경우) 또는 판매하였던 물품은 어떻게 마련했습니까?(판매자인 경우)
            A. {system13}
        14. Q. 재산의 처분은 어떻게 하였습니까?(구매자인 경우) 또는 물품 대금 중 일부도 받지 못했습니까?(판매자인 경우)
            A. {system14}
        15. Q. 거짓말임을 깨닫게 된 계기는 무엇입니까?
            A. {system15}
        16. Q. 다른 피해 사실도 있습니까? 
            A. {system16}
        17. Q. 고소하게 된 동기는 무엇입니까?
            A. {system17}
        18. Q. 사건과 관련하여 민형사를 진행하고 있습니까?
            A. {system18}
        19. Q. 고소장의 수신인은 누구입니까?
            A. {system19}
        20. Q. 고소장의 접수일을 입력하여 주세요.
            A. {system20}
      - Analyze and distribute the information to team members

    2. **Draft Creation** (Jenny)
      - Write an initial draft based on the provided information
      - Use clear and concise language to accurately convey the intent

    3. **Legal Review** (Sam)
      - Review the draft for legal accuracy and effectiveness
      - Modify or add legal terms and s as necessary
      - Identify and suggest alternatives for any legal risks

    4. **Format Review** (Tom)
      - Ensure the structure and format adhere to legal notice standards
      - Verify the accuracy of date, recipient, and sender information
      - Assess the overall readability and professionalism of the document

    5. **Final Review and Approval** (Lisa)
      - Review the entire document for consistency, accuracy, and effectiveness
      - Direct any additional modifications if needed
      - Approve the final document

    ## Important Notes
    - All information must be accurate and fact-based
    - Maintain an objective tone, avoiding emotional s
    - Use legal terminology appropriately, explaining when necessary for layperson understanding
    - Clearly communicate the purpose and requirements of the document
    - Specify any deadlines or conditions explicitly

    ## Final Output Format

    ```
    수신: {system19}
    발신: {system00}
    날짜: {system20}

    제목: 고소장

    [Main Body]

    - Key facts
    - 고소의 취지
    - 범죄 사실
    - 고소 이유
    
    [Conclusion and guidance on further actions]

    [Sender's Signature]
    ```

    ## Language Requirement

    All responses and the final document must be written in Korean.

    ## Examples

    Example 1

    ```
    3. 고소취지
       - 고소인은 피고소인을 중고거래 사기죄로 고소하오니 처벌하여 주시기 바랍니다.

    4. 범죄사실
       - 피고소인은 경호원 이고 고소인은 경비원 입니다. 
       - 2019-09-04경 번개장터에서 고소인은 구매자로써, 양복을 판매 한다, 라고 게시한 피고소인의 글을 보고 피고소인에게 연락을 하여 알게 되었습니다. 
       - 당시 피고소인은 정식 회원으로  등록 되어 있었습니다. 
       - 피고소인의 글은 양복과 넥타이를 판매 한다, 라는 내용으로 되어 있 었습니다. 
       - 고소인이 피고소인의 계좌로 거래대금을 송금하면 피고소인은 거래 물품을 고소인의 자택 주소로 발송해 주기로 하였습니다.
       - 피고소인은 물품대금을 입금받고 물품을 발송하지 않았습니다.
       - 고소인은 월급으로 물품 대금을 마련 하였습니다.
       - 그리고 고소인은 피고소인의 계좌로 30만원을 송금 하였습니다.
       - 그런데 고소인은 피고소인이 물품대금 입금받고 자신의 전화를 차단하여 사기임을 깨닫게 되었습니다.
       - 고소인은 다른 피해는 없습니다.
       
    5. 고소이유
       - 이에 고소인은 피해금액을 돌려 받기 위해 고소를 결심하게 되었 습니다.
    ```

    Example 2

    ```
    3. 고소취지 
       - 고소인은 피고소인을 중고거래 사기죄로 고소하오니 처벌하여 주시기 바랍니다.

    4. 범죄사실
       - 피고소인은 휴대폰 판매점 직원 이고 고소인은 대학생입니다. 
       - 2019-07-06경 번개장터에서 고소인은 구매자로써, 갤럭시 스마트폰 30만원에 판매한다, 라고 게시한 피고소인의 글을 보고 피고소인에게 연락을 하여 알게 되었습니다. 
       - 당시 피고소인은 정식 회원으로  등록 되어 있었습니다. 
       - 피고소인의 글은 갤럭시 스마트폰 30만원에 판매한다, 라는 내용으 로 되어 있었습니다. 
       - 고소인이 거래대금을 피고소인이 알려준 계좌로 송금하면 피고소인 은 거래 물품을 고소인의 주소지로 퀵 서비스를 통해 발송해 주기로 하였습니다.
       - 피고소인은 고소인이 구매한것과 다른 상품을 발송하였습니다.
       - 고소인은 예금으로 물품 대금을 마련 하였습니다.
       - 그리고 고소인은 피고소인이 알려준 계좌로 30만원을 입금 하였습 니다.
       - 그런데 고소인은 피고소인이 환불을 거부하여 사기임을 깨닫게 되었습니다.
       - 고소인은 다른 피해는 없습니다.

    5. 고소이유
       - 이에 고소인은 피해금액을 돌려 받고 피고소인의 처벌을 원하여 고 소를 결심 하게 되었습니다.
    ```

    Follow this prompt to perform your role-specific tasks, collaborating to produce professional and effective legal notices in Korean, similar to the provided examples.

    Team Collaboration Guidelines:
    - Lisa (Team Leader): Oversee the entire process and ensure all team members are working efficiently.
    - Jenny (Document Drafter): Use the examples as a guide for structure and tone when creating the initial draft.
    - Sam (Legal Expert): Pay close attention to the legal terminology and requirements demonstrated in the examples.
    - Tom (Format Reviewer): Ensure the final document follows the format and style shown in the examples.

    Remember to adapt the content to the specific situation while maintaining the professional tone and structure demonstrated in these examples.
  """ 
  prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
  ])
  
  template = []  
  template += [{'text': system_template}]  

  prompt += HumanMessagePromptTemplate.from_template(template=template)
  
  return prompt | llm | StrOutputParser()

def case_retrieval(recipient, sender, date, case):
  return {
    'Recipient': recipient,
    'Sender': sender,
    'Date': date,
    'Subject': case,
    'Content': '\n'.join(d.page_content for d in case_search(case))
  }
  
def result(case):
  return case_retrieval(case)
  
# Title of the web app
st.title("고소장 생성")
st.write("---")

# Text input
system00 = st.text_input("안녕하세요? 이름을 알려주세요.")
system01 = st.text_input("어떤 사기 피해를 입으셨나요?")
system02 = st.text_input("피고소인의 직업이나 신분에 대해 알려주세요.")
system03 = st.text_input("고객님의 신분도 알려주세요.")
system04 = st.text_input("사건이 일어난 일시는 언제인가요?")
system05 = st.text_input("사건이 일어난 장소는 어디인가요?")
system06 = st.text_input("고객께서는 판매자입니까 구매자입니까?")
system07 = st.text_input("처음에 피고소인(들)이 어떻게 접근했습니까? (피고소인을 알게 된 경위)")
system08 = st.text_input("어떤 중고거래 플랫폼을 사용했습니까?")
system09 = st.text_input("피고소인들은 실명을 인증하고 휴대폰 번호가 등록된 정식 회원이었습니까?")
system10 = st.text_input("피고소인이 무엇을 판매한다고 하던가요?(구매자인 경우) 또는 고객께서는 무엇을 판매하려고 어떤 글을 올렸습니까?(판매자인 경우)")
system11 = st.text_input("어떤 방식으로 상품을 받고 돈을 지불하는 걸로 결정했었나요?")
system12 = st.text_input("피고소인 뭐라고 거짓말을 하여 고소인을 속이던가요?")
system13 = st.text_input("재산은 어떻게 마련했습니까?(구매자인 경우) 또는 판매하였던 물품은 어떻게 마련했습니까?(판매자인 경우)")
system14 = st.text_input("재산의 처분은 어떻게 하였습니까?(구매자인 경우) 또는 물품 대금 중 일부도 받지 못했습니까?(판매자인 경우)")
system15 = st.text_input("거짓말임을 깨닫게 된 계기는 무엇입니까?")
system16 = st.text_input("다른 피해 사실도 있습니까?")
system17 = st.text_input("고소하게 된 동기는 무엇입니까?")
system18 = st.text_input("사건과 관련하여 민형사를 진행하고 있습니까?")
system19 = st.text_input("고소장의 수신인은 누구입니까?")
system20 = st.text_input("고소장의 접수일을 입력하여 주세요.")

if st.button('생성하기'):
  with st.spinner('생성하는 중...'):
    # case = case_retrieval(recipient, sender, date, title)
    chain = make_document(
      system00,
      system01,
      system02,
      system03,
      system04,
      system05,
      system06,
      system07,
      system08,
      system10,
      system11,
      system12,
      system13,
      system14,
      system15,
      system16,
      system17,
      system18,
      system19,
      system20
    ) 
    
    query = f'''
      0. Q. 안녕하세요? 이름을 알려주세요.
          A. {system00},
      1. Q. 어떤 사기 피해를 입으셨나요?
          A. {system01},
      2. Q. 피고소인의 직업이나 신분에 대해 알려주세요.
          A. {system02},
      3. Q. 고객님의 신분도 알려주세요.
          A. {system03},
      4. Q. 사건이 일어난 일시는 언제인가요? 
          A. {system04},
      5. Q. 사건이 일어난 장소는 어디인가요?
          A. {system05},
      6. Q. 고객께서는 판매자입니까 구매자입니까?
          A. {system06},
      7. Q. 처음에 피고소인(들)이 어떻게 접근했습니까? (피고소인을 알게 된 경위)
          A. {system07},
      8. Q. 어떤 중고거래 플랫폼을 사용했습니까?
          A. {system08},
      9. Q. 피고소인들은 실명을 인증하고 휴대폰 번호가 등록된 정식 회원이었습니까?
          A. {system09},
      10. Q. 피고소인이 무엇을 판매한다고 하던가요?(구매자인 경우) 또는 고객께서는 무엇을 판매하려고 어떤 글을 올렸습니까?(판매자인 경우)
          A. {system10},
      11. Q. 어떤 방식으로 상품을 받고 돈을 지불하는 걸로 결정했었나요?
          A. {system11},
      12. Q. 피고소인 뭐라고 거짓말을 하여 고소인을 속이던가요?
          A. {system12},
      13. Q. 재산은 어떻게 마련했습니까?(구매자인 경우) 또는 판매하였던 물품은 어떻게 마련했습니까?(판매자인 경우)
          A. {system13},
      14. Q. 재산의 처분은 어떻게 하였습니까?(구매자인 경우) 또는 물품 대금 중 일부도 받지 못했습니까?(판매자인 경우)
          A. {system14},
      15. Q. 거짓말임을 깨닫게 된 계기는 무엇입니까?
          A. {system15},
      16. Q. 다른 피해 사실도 있습니까? 
          A. {system16},
      17. Q. 고소하게 된 동기는 무엇입니까?
          A. {system17},
      18. Q. 사건과 관련하여 민형사를 진행하고 있습니까?
          A. {system18},
      19. Q. 고소장의 수신인은 누구입니까?
          A. {system19}
      20. Q. 고소장의 접수일을 입력하여 주세요.
          A. {system20}
    '''
    
    response = chain.invoke({"query": query + "에 대한 정보를 바탕으로 예시와 같은 고소장을 만들어 줘!"})
    
    st.write(response)