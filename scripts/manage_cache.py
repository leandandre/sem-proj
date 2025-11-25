"""
Manage preprocessing cache.
"""
from sem_proj.data.cache import get_cache_info, clear_cache

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manage preprocessing cache")
    parser.add_argument('--info', action='store_true', help="Show cache info")
    parser.add_argument('--clear', action='store_true', help="Clear all cache")
    parser.add_argument('--subject', type=str, help="Clear cache for specific subject")
    
    args = parser.parse_args()
    
    if args.info:
        info = get_cache_info()
        print("="*60)
        print("CACHE INFO")
        print("="*60)
        print(f"Number of files: {info['num_files']}")
        print(f"Total size: {info['total_size_mb']:.2f} MB")
        print(f"Subjects cached: {len(info['subjects'])}")
        print("="*60)
    
    elif args.clear:
        if args.subject:
            print(f"Clearing cache for {args.subject}...")
            clear_cache(subject=args.subject)
        else:
            confirm = input("Clear ALL cache? (yes/no): ")
            if confirm.lower() == 'yes':
                clear_cache()
            else:
                print("Cancelled")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()