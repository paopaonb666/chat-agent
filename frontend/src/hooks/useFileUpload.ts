import { useState, useCallback } from 'react';
import type { UploadedFile } from '../types';
import { uploadFile } from '../services/api';

export function useFileUpload() {
  const [uploading, setUploading] = useState(false);

  const uploadFiles = useCallback(async (convId: string, files: FileList): Promise<UploadedFile[]> => {
    setUploading(true);
    try {
      const results: UploadedFile[] = [];
      for (const file of Array.from(files)) {
        try {
          results.push(await uploadFile(convId, file));
        } catch {
          // skip failed uploads
        }
      }
      return results;
    } finally {
      setUploading(false);
    }
  }, []);

  return { uploadFiles, uploading };
}
