import fetch_comments
import visualize_data

if __name__ == '__main__':
    fetch_comments.grab_comments_from_cloud()
    visualize_data.visualize_topics(True,8)
