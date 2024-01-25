from re import A
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import boto3
from streamlit_folium import st_folium      # streamlitでfoliumを使う
import folium
import math     

def initialze_page():
    st.set_page_config(
        page_title = "Hotel Recommend Tools",
        page_icon = "🏨"
    )

def aws_embedding(texts): #list
    bedrock = boto3.client('bedrock-runtime',  region_name='us-west-2')
    
    params = {
      "modelId": "cohere.embed-multilingual-v3",
      "contentType": "application/json",
      "accept": "*/*",
      "body": json.dumps({
        "texts": texts,
        "input_type": "search_document"
      })
    }
    
    response = bedrock.invoke_model(**params)
    
    body = response['body'].read().decode('utf-8')
    json_body = json.loads(body)
    return json_body['embeddings'][0]

def cos_similarity(a,b):
    return np.dot(a,b) / ((np.sqrt(np.dot(a,a))) * (np.sqrt(np.dot(b,b))))

def nearby_hotels(df:pd.DataFrame, ido:float, keido:float):
    #近隣ホテルの定義: 当該ホテルを中心とし、半径が緯度3分、経度4分の長方形
    keido_lim = 4/60 
    ido_lim = 3/60 #度に変換
    df = df[df['ido'] <= ido+ido_lim]
    df = df[df['ido'] >= ido-ido_lim]
    df = df[df['keido'] <= keido+keido_lim]
    df = df[df['keido'] >= keido-keido_lim]
    df['dis_score'] = np.zeros(len(df))
    df = df.reset_index(drop=True)
    for i in range(len(df)):
        distance = np.sqrt(abs(df['ido'][i] - ido)**2 + abs(df['keido'][i] - keido)**2)
        score = (np.sqrt(keido_lim**2 + ido_lim**2) - distance)/(np.sqrt(keido_lim**2 + ido_lim**2))
        if score < 0:
            score = 0
        df['dis_score'][i] = score
    df['total_score'] = df['sim'] + df['dis_score']*0.1
    df = df.sort_values('total_score', ascending=False)[:5]

    return df

def nearby_pop(df:pd.DataFrame, ido:float, keido:float):
    keido_lim = 8/60
    ido_lim = 6/60
    df = df[df['ido'] <= ido+ido_lim]
    df = df[df['ido'] >= ido-ido_lim]
    df = df[df['keido'] <= keido+keido_lim]
    df = df[df['keido'] >= keido-keido_lim]
    df = df[df['hotelid'].notnull()]
    df = df[df['hotelid'] != '']
    df = df.reset_index(drop=True)
    database = pd.read_csv('./KNTres20230927-20240109.csv', encoding='shift-jis')['hotelid'].to_list()
    gacount = []
    ids = df['hotelid'].to_list()
    for id in ids:
        gacount.append(database.count(id))
    df['gacount'] = gacount
    df = df.sort_values('gacount', ascending=False).reset_index(drop=True)[:5]
    return df

def nearby_pop2(df:pd.DataFrame, ido:float, keido:float):
    keido_lim = 3/60
    ido_lim = 2/60
    df = df[df['ido'] <= ido+ido_lim]
    df = df[df['ido'] >= ido-ido_lim]
    df = df[df['keido'] <= keido+keido_lim]
    df = df[df['keido'] >= keido-keido_lim]
    df = df[df['hotelid'].notnull()]
    df = df[df['hotelid'] != '']
    df = df.reset_index(drop=True)
    database = pd.read_csv('./KNTres20230927-20240109.csv', encoding='shift-jis')['hotelid'].to_list()
    gacount = []
    ids = df['hotelid'].to_list()
    for id in ids:
        gacount.append(database.count(id))
    df['gacount'] = gacount
    df = df.sort_values('gacount', ascending=False).reset_index(drop=True)
    return df

def limit_price(df:pd.DataFrame, dprice:int):
    if dprice < 20000:
        p_uplim = dprice+20000
    elif dprice < 50000:
        p_uplim = 1.5*dprice+10000
    else:
        p_uplim = 9999999
    
    if dprice < 50000:
        p_lolim = dprice*0.75 - 10000
    elif dprice < 100000:
        p_lolim = dprice*0.4 + 8000
    else:
        p_lolim = 48000

    df = df[df['price'] <= p_uplim]
    df = df[df['price'] >= p_lolim]

    return df
    
def main():
    st.title("ホテル検索ツール")
    df = pd.read_pickle('vector_database.pkl')
    df = df.reset_index(drop=True)
    st.session_state['df'] = df
    search_word = st.text_input('地名・設備・ホテルの特徴などで検索', placeholder = "例：東北の自然に囲まれた温泉宿", key = "search_word")
    pressed = st.button("Search Hotels")
    df['sim'] = np.zeros(len(df))

    if pressed:
        search_vec = aws_embedding([search_word])
        for i in range(len(df)):
            df['sim'][i] = cos_similarity(search_vec, df['embedding'][i])
        df = df.sort_values('sim', ascending=False)
        st.session_state['df'] = df
        display = df.iloc[0:10].reset_index(drop=True)
        def change_page():
            st.session_state["page-select"] = "page2"

        def button_callback(n:int):
            st.session_state["name"] = display['name'][n]
            st.session_state["title"] = display['title'][n]
            st.session_state["content"] = display['content'][n]
            st.session_state["embedding"] = display['embedding'][n]
            st.session_state["ido"] = display['ido'][n]
            st.session_state["keido"] = display['keido'][n]
            st.session_state["url"] = display['url'][n]
            st.session_state['hotelid'] = display['hotelid'][n]
            st.session_state['price'] = display['price'][n]
            change_page()

        for i in range(len(display)):
            st.session_state["url"] = display['url'][i]
            st.session_state["title"] = display['title'][i]
            st.session_state["content"] = display['content'][i]
            name, price, content = display['name'][i], display['price'][i], display['content'][i]
            st.markdown(f'**{name}**  \n{price}円～  \n{content}')
            st.button(f"{name}の詳細", on_click=button_callback, args=(i,))
    
    st.header("おすすめホテル")
    pref = st.selectbox(
        'お住いの都道府県を選択 (実際のサイトでは利用者の位置情報を基に表示します)',
        ('北海道','青森県','岩手県','宮城県','秋田県','山形県','福島県','茨城県','栃木県','群馬県','埼玉県','千葉県','東京都','神奈川県','新潟県','富山県','石川県','福井県','山梨県','長野県','岐阜県','静岡県','愛知県','三重県','滋賀県','京都府','大阪府','兵庫県','奈良県','和歌山県','鳥取県','島根県','岡山県','広島県','山口県','徳島県','香川県','愛媛県','高知県','福岡県','佐賀県','長崎県','熊本県','大分県','宮崎県','鹿児島県','沖縄県','海外' )
    )

    
    hotel = pd.read_csv('./KNTres20230927-20240109.csv', encoding='shift-jis')
    pref_df = pd.read_csv('./pref.csv', encoding='shift-jis')
    pref_eng = pref_df[pref_df['name']==pref]['en'].iloc[0]
    hotel_pref = hotel[hotel['pref']==pref_eng]
    top_hotels = list(hotel_pref['hotelid'].value_counts()[0:10].index)
    results = df[df['hotelid'].isin(top_hotels)].reset_index(drop=True)

    def change_page():
        st.session_state["page-select"] = "page2"

    def button_callback(n:int):
        st.session_state["name"] = results['name'][n]
        st.session_state["title"] = results['title'][n]
        st.session_state["content"] = results['content'][n]
        st.session_state["embedding"] = results['embedding'][n]
        st.session_state["ido"] = results['ido'][n]
        st.session_state["keido"] = results['keido'][n]
        st.session_state["url"] = results['url'][n]
        st.session_state['hotelid'] = results['hotelid'][n]
        st.session_state['price'] = results['price'][n]
        change_page()

    for i in range(len(results)):
        st.session_state["url"] = results['url'][i]
        st.session_state["title"] = results['title'][i]
        st.session_state["content"] = results['content'][i]
        name, price, content = results['name'][i], results['price'][i], results['content'][i]
        st.markdown(f'**{name}**  \n{price}円～  \n{content}')
        st.button(f"{name}の詳細", on_click=button_callback, args=(i,))

def detail():
    st.title(st.session_state["name"])
    dprice = st.session_state['price']
    st.write(f"{dprice}円～")
    if st.session_state['hotelid'] != '':
        st.write("近畿日本ツーリストで予約 (%s)" % f"https://yado.knt.co.jp/planlist/{st.session_state['hotelid']}/")


    st.write(st.session_state["content"])
    df_rank = st.session_state['df']

    df_rank = limit_price(df_rank, dprice)
    
    df_rank = df_rank.reset_index(drop=True)
    df_rank['rank'] = np.zeros(len(df_rank))
    for i in range(len(df_rank)):
        df_rank['sim'][i] = cos_similarity(df_rank['embedding'][i], st.session_state["embedding"])
    df_rank = nearby_hotels(df_rank, st.session_state["ido"], st.session_state["keido"])
    df_rank = pd.concat([df_rank, nearby_pop(limit_price(st.session_state['df'],dprice),st.session_state['ido'],st.session_state['keido'])])
    df_rank = df_rank.drop_duplicates(subset='name')

    def button_callback(n:int):
            st.session_state["name"] = df_rank['name'][n]
            st.session_state["title"] = df_rank['title'][n]
            st.session_state["content"] = df_rank['content'][n]
            st.session_state["embedding"] = df_rank['embedding'][n]
            st.session_state["ido"] = df_rank['ido'][n]
            st.session_state["keido"] = df_rank['keido'][n]
            st.session_state["url"] = df_rank['url'][n]
            st.session_state['hotelid'] = df_rank['hotelid'][n]
            st.session_state['price'] = df_rank['price'][n]

    lat = st.session_state["ido"]
    long = st.session_state["keido"]
    df_rank = df_rank.reset_index(drop=True)

    # 地図の中心の緯度/経度、タイル、初期のズームサイズを指定します。
    m = folium.Map(
        # 地図の中心位置の指定(今回は栃木県の県庁所在地を指定)
        location=[lat, long], 
        # タイル、アトリビュートの指定
        tiles='https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png',
        attr='ホテルマップ',
        # ズームを指定
        zoom_start=14
    )

    # 読み込んだデータ(緯度・経度、ポップアップ用文字、アイコンを表示)
    for i, row in df_rank.iterrows():
        # ポップアップの作成(都道府県名＋都道府県庁所在地＋人口＋面積)
        pop=f"{row['name']}"
        folium.Marker(
                # 緯度と経度を指定
                location=[row['ido'], row['keido']],
                # ツールチップの指定(都道府県名)
                tooltip=row['name'],
                # ポップアップの指定
                popup=folium.Popup(pop, max_width=300),
                # アイコンの指定(アイコン、色)
                icon=folium.Icon(icon="home",icon_color="white", color="red")
        ).add_to(m)
    st_data = st_folium(m, width=700,height=800)

    
    #コサイン類似度ではなく人気度順で表示した場合の半径5kmおすすめ。どちらのがよいだろうか？
    #pop2 = nearby_pop2(st.session_state['df'],st.session_state['ido'],st.session_state['keido'])
    #pop2 = limit_price(pop2,dprice)
    #st.dataframe(pop2.reset_index(drop=True)[:5])
    

    st.header("近隣のおすすめホテル")

    for i in range(1, len(df_rank)):
        name,content, price = df_rank['name'][i], df_rank['content'][i], df_rank['price'][i]
        st.markdown(f'**{name}**  \n{price}円～  \n{content}')
        st.button(f"{name}の詳細", on_click=button_callback, args=(i,))
        
    
    
    transition_df = pd.read_pickle('./transition_data.pkl')
   
    suggestable = 0
    if len(transition_df[transition_df['hotelid']==st.session_state['hotelid']]['rank']) != 0:
        transition_list = transition_df[transition_df['hotelid']==st.session_state['hotelid']]['rank'].iloc[0]
        if len(transition_list) == 0:
            suggestable = 1
    else:
        suggestable = 1


    if suggestable == 0:
        st.header("このホテルを見た人はこんなホテルも見ています")
        suggest = st.session_state['df']

        suggest = limit_price(suggest, dprice)
        namelist = suggest['name'].to_list()
        contentlist = suggest['content'].to_list()
        idlist = suggest['hotelid'].to_list()
        pricelist = suggest['price'].to_list()

        def callback2(n:int):
            st.session_state["name"] = suggest['name'][n]
            st.session_state["title"] = suggest['title'][n]
            st.session_state["content"] = suggest['content'][n]
            st.session_state["embedding"] = suggest['embedding'][n]
            st.session_state["ido"] = suggest['ido'][n]
            st.session_state["keido"] = suggest['keido'][n]
            st.session_state["url"] = suggest['url'][n]
            st.session_state['hotelid'] = suggest['hotelid'][n]
            st.session_state['price'] = suggest['price'][n]


        for i in range(len(transition_list)): 
            try:    
                id = transition_list[i][0]
                idx = idlist.index(id)
                name = namelist[idx]
                content = contentlist[idx]
                price = pricelist[idx]
                st.markdown(f'**{name}**  \n{price}円～  \n{content}')
                st.button(f'{name} 詳細', on_click=callback2, args=(idx,))
            except:
                continue

    def return_home():
        st.session_state["page-select"] = "page1"
    st.button("ホームに戻る", on_click=return_home)

pages = dict(
    page1="検索",
    page2="詳細",
)

page_id = st.sidebar.selectbox(
    "ページ名",
    ["page1", "page2"],
    format_func=lambda page_id: pages[page_id],
    key = "page-select",
)

if page_id == "page1":
    main()

if page_id == "page2":
    detail()