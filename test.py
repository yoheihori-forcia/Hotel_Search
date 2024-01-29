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
import collections

def initialze_page():
    st.set_page_config(
        page_title = "Hotel Recommend Tools",
        page_icon = "🏨"
    )

def aws_embedding(texts:list): #list
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

    dis_score = []
    idos = df['ido'].to_list()
    keidos = df['keido'].to_list()
    idokeido = zip(idos,keidos)
    df = df.reset_index(drop=True)

    for i, k in idokeido:
        distance = np.sqrt(abs(i-ido)**2 + abs(k-keido)**2)
        score = (np.sqrt(keido_lim**2 + ido_lim**2) - distance)/(np.sqrt(keido_lim**2+ido_lim**2))
        if score < 0:
            score = 0
        dis_score.append(score)

    df['dis_score'] = dis_score
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
    
def df_to_lists(display:pd.DataFrame):
    urls = display['url'].to_list()
    titles = display['title'].to_list()
    contents =display['content'].to_list()
    names = display['name'].to_list()
    embeddings = display['embedding'].to_list()
    idos = display['ido'].to_list()
    keidos = display['keido'].to_list()
    hotelids = display['hotelid'].to_list()
    prices = display['price'].to_list()

    return urls,titles,contents,names,embeddings,idos,keidos,hotelids,prices

def personalize(gaid:str,df:pd.DataFrame,history:pd.DataFrame):
    history_df =history[history['GA']==gaid]
    history_p = history_df['hotelid'].to_list()
    ids = df['hotelid'].to_list()
    embeddings = df['embedding'].to_list()
    personal_v= np.zeros(1024).tolist()
    pref = []
    for id in history_p:
        if id in ids:
            personal_v = [x+y for x,y in zip(personal_v,embeddings[ids.index(id)])]
            pref.append(id[1:3])
    pref_most = collections.Counter(pref).most_common()

    return personal_v, pref_most, history_df

def add_vector():
    st.session_state['personal_v'] = [x+y for x,y in zip(st.session_state['personal_v'],st.session_state['embedding'])]



def main():
     #パーソナル情報初期設定
    @st.cache_data
    def load_vdb():
        return pd.read_pickle('vector_database.pkl')
    df = load_vdb()

    if 'personal_v' not in st.session_state:
        @st.cache_data
        def read_res():
            return pd.read_csv('./KNTres20230927-20240109.csv', encoding='shift-jis')
        history = read_res()
        gaid = history['GA'].unique()[np.random.randint(history['GA'].nunique())]
        st.session_state['gaid'] = gaid
        st.session_state['personal_v'], st.session_state['personal_pref'], st.session_state['personal_history'] = personalize(gaid,df,history)
    else:
        gaid = st.session_state['gaid']

    st.caption(f"ようこそ、{gaid} さん")
    st.title("ホテル検索ツール")


    df = df.sort_values('gacount')
    df = df.reset_index(drop=True)
    st.session_state['df'] = df
    if 'search_word' not in st.session_state:
        st.session_state.search_word = ''
    search_word = st.text_input('地名・設備・ホテルの特徴などで検索', placeholder = "例：東北の自然に囲まれた温泉宿", key = "search_word", value=st.session_state['search_word'])
    pressed = st.button("Search Hotels")
    df['sim'] = np.zeros(len(df))

    #検索結果
    if pressed:
        search_vec = aws_embedding([search_word])
        sim = []
        embeddings = df['embedding'].to_list()
        for embedding in embeddings:
            sim.append(cos_similarity(search_vec, embedding))
        df['sim'] = sim
        df = df.sort_values('gacount', ascending=False).head(int(len(df)*0.2))
        df = df.sort_values('sim', ascending=False)
        display = df.iloc[0:10]
        display = display.sort_values("blueplanet", ascending=False).sort_values("gacount", ascending=False)
        def change_page():
            st.session_state["page-select"] = "page2"
            add_vector()

        def button_callback(n:int):
            st.session_state["name"] = names[n]
            st.session_state["title"] = titles[n]
            st.session_state["content"] = contents[n]
            st.session_state["embedding"] = embeddings[n]
            st.session_state["ido"] = idos[n]
            st.session_state["keido"] = keidos[n]
            st.session_state["url"] = urls[n]
            st.session_state['hotelid'] = hotelids[n]
            st.session_state['price'] = prices[n]
            change_page()
            

        urls,titles,contents,names,embeddings,idos,keidos,hotelids,prices = df_to_lists(display)

        for i in range(len(display)):
            st.session_state["url"] = urls[i]
            st.session_state["title"] = titles[i]
            st.session_state["content"] = contents[i]
            name, price, content = names[i], prices[i], contents[i]
            st.markdown(f'**{name}**  \n{price}円～  \n{content}')
            st.button(f"{name}の詳細", on_click=button_callback, args=(i,))
    
    st.header("位置情報おすすめホテル")
    pref = st.selectbox(
        'お住いの都道府県を選択 (実際のサイトでは利用者の位置情報を基に表示します)',
        ('北海道','青森県','岩手県','宮城県','秋田県','山形県','福島県','茨城県','栃木県','群馬県','埼玉県','千葉県','東京都','神奈川県','新潟県','富山県','石川県','福井県','山梨県','長野県','岐阜県','静岡県','愛知県','三重県','滋賀県','京都府','大阪府','兵庫県','奈良県','和歌山県','鳥取県','島根県','岡山県','広島県','山口県','徳島県','香川県','愛媛県','高知県','福岡県','佐賀県','長崎県','熊本県','大分県','宮崎県','鹿児島県','沖縄県',)
    )

    df = st.session_state['df']
    @st.cache_data
    def pref_pop(pref):
        @st.cache_data
        def read_res():
            return pd.read_csv('./KNTres20230927-20240109.csv', encoding='shift-jis')
        hotel = read_res()
        @st.cache_data
        def read_pref():
            return pd.read_csv('./pref.csv', encoding='shift-jis')
        pref_df = read_pref()
        pref_eng = pref_df[pref_df['name']==pref]['en'].iloc[0]
        hotel_pref = hotel[hotel['pref']==pref_eng]
        top_hotels = list(hotel_pref['hotelid'].value_counts()[0:10].index)
        results = df[df['hotelid'].isin(top_hotels)].sort_values(["blueplanet","gacount"], ascending=False).reset_index(drop=True)
        ids = results['hotelid'].to_list()
        ppref = []
        for id in ids:
            if id[1:3] == st.session_state.personal_pref:
                ppref.append(1)
            else:
                ppref.append(0)

        results['ppref'] = ppref
        results = results.sort_values('ppref', ascending=False).reset_index(drop=True)
        return results
    
    results = pref_pop(pref)

    def change_page():
        st.session_state["page-select"] = "page2"
        add_vector()

    def button_callback2(n:int):
        st.session_state["name"] = names2[n]
        st.session_state["title"] = titles2[n]
        st.session_state["content"] = contents2[n]
        st.session_state["embedding"] = embeddings2[n]
        st.session_state["ido"] = idos2[n]
        st.session_state["keido"] = keidos2[n]
        st.session_state["url"] = urls2[n]
        st.session_state['hotelid'] = hotelids2[n]
        st.session_state['price'] = prices2[n]
        change_page()

    urls2,titles2,contents2,names2,embeddings2,idos2,keidos2,hotelids2,prices2 = df_to_lists(results)

    for i in range(len(results)):
        st.session_state["url"] = urls2[i]
        st.session_state["title"] = titles2[i]
        st.session_state["content"] = contents2[i]
        name, price, content = names2[i], prices2[i], contents2[i]
        st.markdown(f'**{name}**  \n{price}円～  \n{content}')
        st.button(f"{name}の詳細", on_click=button_callback2, args=(i,))


    st.header("あなたへのおすすめホテル")
    sim = []
    embeddings = st.session_state.df['embedding'].to_list()
    for embedding in embeddings:
        sim.append(cos_similarity(st.session_state['personal_v'], embedding))
    personal = st.session_state.df
    personal['sim'] = sim
    personal = personal.sort_values('gacount', ascending=False).head(int(len(df)*0.2))
    personal = personal.sort_values('sim', ascending=False)
    personal = personal.iloc[0:10].reset_index(drop=True)
    hotelidsp = personal['hotelid'].to_list()
    ppref = []
    for id in hotelidsp:
        if id[1:3] == st.session_state.personal_pref:
            ppref.append(1)
        else:
            ppref.append(0)
    personal['ppref'] = ppref
    personal = personal.sort_values('ppref', ascending=False).reset_index(drop=True)
    #personal = personal.sort_values("blueplanet", ascending=False).sort_values("gacount", ascending=False)


    def change_page():
        st.session_state["page-select"] = "page2"
        add_vector()

    def button_callbackp(n:int):
        st.session_state["name"] = namesp[n]
        st.session_state["title"] = titlesp[n]
        st.session_state["content"] = contentsp[n]
        st.session_state["embedding"] = embeddingsp[n]
        st.session_state["ido"] = idosp[n]
        st.session_state["keido"] = keidosp[n]
        st.session_state["url"] = urlsp[n]
        st.session_state['hotelid'] = hotelidsp[n]
        st.session_state['price'] = pricesp[n]
        change_page()
            

    urlsp,titlesp,contentsp,namesp,embeddingsp,idosp,keidosp,hotelidsp,pricesp = df_to_lists(personal)

    for i in range(len(personal)):
        st.session_state["url"] = urlsp[i]
        st.session_state["title"] = titlesp[i]
        st.session_state["content"] = contentsp[i]
        name, price, content = namesp[i], pricesp[i], contentsp[i]
        st.markdown(f'**{name}**  \n{price}円～  \n{content}')
        st.button(f"{name} の詳細", on_click=button_callbackp, args=(i,))

    def move_to_history():
        st.session_state["page-select"] = "page3"

    st.button("予約履歴へ", on_click=move_to_history)

    



def detail():
    #当該ホテル詳細
    st.title(st.session_state["name"])
    dprice = st.session_state['price']
    st.write(f"{dprice}円～")
    if st.session_state['hotelid'] != '':
        st.write("近畿日本ツーリストで予約 (%s)" % f"https://yado.knt.co.jp/planlist/{st.session_state['hotelid']}/")

    st.write(st.session_state["content"])


    #おすすめホテル抽出
    df_rank = st.session_state['df']
    df_rank = limit_price(df_rank, dprice)

    df_rank = df_rank.reset_index(drop=True)
    df_rank['rank'] = np.zeros(len(df_rank))
    sims = []
    embeddings = df_rank['embedding'].to_list()
    for embedding in embeddings:
        sims.append(cos_similarity(embedding, st.session_state["embedding"]))
    df_rank['sim'] = sims
    df_rank = nearby_hotels(df_rank, st.session_state["ido"], st.session_state["keido"])
    df_rank = pd.concat([df_rank, nearby_pop(limit_price(st.session_state['df'],dprice),st.session_state['ido'],st.session_state['keido'])])
    df_rank = df_rank.drop_duplicates(subset='name')[1:].sort_values("blueplanet", ascending=False).reset_index(drop=True)


    #地図描画
    lat = st.session_state["ido"]
    long = st.session_state["keido"]

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
    

    #おすすめホテル表示
    st.header("近隣のおすすめホテル")

    urls,titles,contents,names,embeddings,idos,keidos,hotelids,prices = df_to_lists(df_rank)
    def change_page():
        st.session_state["page-select"] = "page2"
        add_vector()

    def button_callback(n:int):
        st.session_state["name"] = names[n]
        st.session_state["title"] = titles[n]
        st.session_state["content"] = contents[n]
        st.session_state["embedding"] = embeddings[n]
        st.session_state["ido"] = idos[n]
        st.session_state["keido"] = keidos[n]
        st.session_state["url"] = urls[n]
        st.session_state['hotelid'] = hotelids[n]
        st.session_state['price'] = prices[n]
        change_page()

    for i in range(len(df_rank)):
        name, content, price = names[i], contents[i], prices[i]
        st.markdown(f'**{name}**  \n{price}円～  \n{content}')
        st.button(f"{name}の詳細", on_click=button_callback, args=(i,))
        
    @st.cache_data
    def load_tbd():
        return pd.read_pickle('transition_data.pkl')
    transition_df = load_tbd()
   
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

        urls2,titles2,contents2,names2,embeddings2,idos2,keidos2,hotelids2,prices2 = df_to_lists(suggest)

        def change_page():
            st.session_state["page-select"] = "page2"
            add_vector()

        def button_callback2(n:int):
            st.session_state["name"] = names2[n]
            st.session_state["title"] = titles2[n]
            st.session_state["content"] = contents2[n]
            st.session_state["embedding"] = embeddings2[n]
            st.session_state["ido"] = idos2[n]
            st.session_state["keido"] = keidos2[n]
            st.session_state["url"] = urls2[n]
            st.session_state['hotelid'] = hotelids2[n]
            st.session_state['price'] = prices2[n]
            change_page()


        for i in range(len(transition_list)): 
            try:    
                id = transition_list[i][0]
                idx = hotelids2.index(id)
                name = names2[idx]
                content = contents2[idx]
                price = prices2[idx]
                st.markdown(f'**{name}**  \n{price}円～  \n{content}')
                st.button(f'{name}  詳細', on_click=button_callback2, args=(idx,))
            except:
                continue

    def return_home():
        st.session_state["page-select"] = "page1"
    st.button("ホームに戻る", on_click=return_home)

def reserve_history():
    st.write(f"{st.session_state.gaid}の予約履歴（2023/9/27～2024/1/9)")
    st.write("検索ページの「あなたへのおすすめ」はこのデータを基にして表示")
    st.dataframe(st.session_state.personal_history)
    



pages = dict(
    page1="検索",
    page2="詳細",
    page3="予約履歴"
)

page_id = st.sidebar.selectbox(
    "ページ名",
    ["page1", "page2", "page3"],
    format_func=lambda page_id: pages[page_id],
    key = "page-select",
)

if page_id == "page1":
    main()

if page_id == "page2":
    detail()

if page_id == "page3":
    reserve_history()