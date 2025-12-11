import lmdb

def test_lmdb_database(lmdb_path):
    """Test LMDB database access and content"""
    try:
        print(f"Testing LMDB: {lmdb_path}")
        
        # Try to open the database
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
        with env.begin() as txn:
            # Get database statistics
            stat = env.stat()
            print(f"Database statistics: {stat}")
            print(f"Total entries: {stat['entries']}")
            
            # Try to read first few entries
            cursor = txn.cursor()
            count = 0
            for key, value in cursor:
                print(f"Entry {count}: key={key}, value_size={len(value)} bytes")
                count += 1
                if count >= 5:  # Show first 5 entries
                    break
        
        env.close()
        return True
        
    except Exception as e:
        print(f"Error accessing LMDB: {e}")
        return False

# Test your actual LMDB file
# Replace with the actual path you find
test_lmdb_database("/media/chge7185/HDD1/repositories/lmdb_sa1b/sa1b_streaming.lmdb")
