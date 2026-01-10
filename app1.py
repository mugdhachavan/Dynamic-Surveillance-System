import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_option_menu import option_menu

st.set_page_config(
    layout="wide",
    page_title="Dynamic Survilliance",
    page_icon="⭐",
)

# option = st.sidebar.radio("Choose your option....", ("About Us", "Video Detection", "Image Detection")
with st.sidebar:
    option = option_menu(
        menu_title="Menu",
        options=["About Us", "Video Detection", "Image Detection"],
        icons=["Bookmarks-fill", "Camera-video-fill", "Image-fill"],
        menu_icon="Menu-button-wide-fill",
        default_index=0,
    )

if option == "About Us":
    # unsolved bg-color
    # st.markdown("""
    # <style>
    # .stappview-container.st-emotion-cache-1wrcr25.ea3mdgi9{
    # background-color: red;
    # }
    # """,unsafe_allow_html=True)
    with stylable_container(
        key="home_title",
        css_styles="""
        h1{
            text-align:center;
            color:purple;
        }
        """
    ):
        st.title("Real-Time Human Insight System")


    with stylable_container(
        key="text1",
        css_styles="""
            .stMarkdown{
                font-family:Optima, sans-serif;
                font-size: 64px;
            }
        """
    ):
        con = st.container(border=True)
        con.markdown('''
            Welcome to our real-time object detection showcase! Explore the capabilities of our system as we detect    
        and track people in live video streams using cutting-edge technology.                                
                                                              
        ****Project Overview:****                                    
        Our project is a real-time object detection system that utilizes state-of-the-art technology to detect    
        and track people in live video streams. We integrate the YOLO object detection model to achieve accurate    
        and reliable detection results. 
        ''')




    # st.subheader("About our Product")
    # st.text_area("",
    #              "Welcome to [Your Project Name], where innovation meets real-world application. Our project is a"
    #              "culmination of passion, technology, and a desire to make a difference in the world of person"
    #              "tracking.",)

    st.image("about.jpg")
    with stylable_container(
            key="text2",
            css_styles="""
            .stMarkdown{
                font-family:Optima, sans-serif;
                font-size: 64px;
            }
        """
    ):
        con2 = st.container(border=True)
        con2.markdown('''
            Detailed Description:                                  
            Our system captures a video stream and applies the YOLO object detection model to detect people in each   
            frame. We use a deque to track detection history and determine the presence of a person. Additionally, we   
            provide real-time feedback on the occupancy status and number of people detected.  
        ''')


    # block - container.st - emotion - cache - gh2jqd.ea3mdgi5

elif option =="Video Detection":
    with stylable_container(
        key="vid_title",
        css_styles="""
        h1{
            color:white;
            font-family:serif;
        }
        """
    ):
        st.title("Real Time Insight System")

    col11,col12,col13=st.columns(3)
    with col11:
        container=st.container(height=200)
        container.title("Recording Functionality")
        container.markdown('''
        Records video clips when a person is detected, 
        enabling further analysis or monitoring.
        ''')

    with col12:
        container=st.container(height=200)
        container.title("Performance Metrics")
        container.markdown('''
        Displays real-time metrics such as Frames Per Second 
        (FPS),area occupancy status, number of people detected, 
        and remaining patience time.
        ''')

    with col13:
        container=st.container(height=200)
        container.title("Customizable Output Path")
        container.markdown('''
         Saves recorded videos to a customizable output path 
         for easy access and management.
        ''')

    col21,col22,col23=st.columns(3)
    with col21:
        container=st.container(height=200)
        container.title("Patience Mechanism")
        container.markdown('''
         Implements a dynamic patience mechanism to minimize 
         false positives and ensure accurate detection status.
        ''')

    with col22:
        container=st.container(height=200)
        container.title("Occupancy Monitoring")
        container.markdown('''
        Tracks the number of people detected in real-time and 
        provides feedback on the occupancy status of the area.
        ''')

    with col23:
        container=st.container(height=200)
        container.title("SMS Notifications")
        container.markdown('''
        Sends SMS messages to your cell phone when a person is 
        detected and when they leave, including timestamps for 
        each event.
        ''')

    add_vertical_space(4)

    col31,col32=st.columns(2)
    with col31:
        with stylable_container(
            key="in_vid",
            css_styles="""
            h1{
                font-family:serif;
                color:red;
                text-align:center;
                padding:35px,35px;
                margin-above:80px;
                margin-below:80px;
            }
            """
        ):
            st.title("Input Video")

    with col32:
        st.video("testvid.mp4")
    add_vertical_space(4)
    col41,col42=st.columns(2)

    with col41:
        st.video("output_video.mp4")

    with col42:
        with stylable_container(
                key="out_vid",
                css_styles="""
                 h1{
                     font-family:serif;
                     color:red;
                     font-size:300px
                     max-width: 100%;
                     height: auto;
                 }
                 """
        ):
            st.title("Output video")


elif option =="Image Detection":
    with stylable_container(
        key="title",
        css_styles="""
        h1{
            color:#F0F8FF;
            margin-left:50px;
            font-family: "Times New Roman", Times, serif;
        }
        """
    ):
        st.title("Object Detection on Image.")
    add_vertical_space()

    col1,col2 = st.columns(2)
    with col1:
        with stylable_container(
            key="heading1",
            css_styles="""
            h2{
                text-align: center;
                color: #8A2BE2;
            }
            """,
        ):
            st.header("Input Image")

        add_vertical_space()
        with stylable_container(
                key="input_img",
                css_styles="""
                img{
                  border-radius:100px;
                  padding: 25px;
                  max-width: 100%; 
                  height: auto;
                }

                """,
        ):
                st.image("undetected.jpg",use_column_width=True)


    with col2:
        with stylable_container(
            key="heading",
            css_styles="""
            h2{
                text-align: center;
                color:#8A2BE2;
            }
            """,
        ):
            st.header("Output Image")

        add_vertical_space()
        with stylable_container(
                key="output_img",
                css_styles="""
                img{
                  border-radius:100px;
                  padding: 25px;
                  max-width: 100%; 
                  height: auto;
                }

                """,
        ):
                st.image("detected.jpg",use_column_width=True)
    # with stylable_container(
    #     key="img_text",
    #     css_styles=["""
    #     {
    #     background-color: #FFF8DC;
    #     }
    #     """
    #     ,
    #     """
    #     p{
    #              font-family:Bookman, URW Bookman L, serif;
    #              color:black;
    #              font-size:25px;
    #             }
    #
    #     """]
    # ):
    #     container = st.container(border=True)
    #     container.markdown('''
    #             Experience the transformation of images through our powerful image processing techniques. Witness
    #             the magic as we enhance, analyze, and detect objects in images with precision.
    #     ''')



