import scripts.download_tinyimagenet as download_script
def main():
    # check if data set is available, if not download it
    dataset_Path=download_script.download_and_extract_tiny_imagenet()
    

if __name__ == '__main__':
    main()