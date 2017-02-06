
#ifndef LABEL_SET_H_
#define LABEL_SET_H_

#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>

#include <set>
#include <vector>

typedef struct {
  std::set<int>* _set;
  pthread_mutex_t lock;
} label_set;

void label_set_insert(label_set s, int label) {
  pthread_mutex_lock(&s.lock);
  s._set->insert(label);
  pthread_mutex_unlock(&s.lock);
}

void label_set_init(label_set* s) {
  s->_set = new std::set<int>();
  pthread_mutex_init(&s->lock, NULL);
}

bool label_set_find(label_set s, int key) {
  pthread_mutex_lock(&s.lock);
  bool temp= s._set->find(key) != s._set->end();
  pthread_mutex_unlock(&s.lock);
  return temp;
}

void label_set_contents(label_set s, std::vector<int>* list) {
  pthread_mutex_lock(&s.lock);
  for(std::set<int>::iterator it = s._set->begin();
      it != s._set->end();) { // note missing it++
    int label = *it;
    list->push_back(label);
    ++it;
  }
  pthread_mutex_unlock(&s.lock);
}

void label_set_remove(label_set s, int key) {
  pthread_mutex_lock(&s.lock);
  if (s._set->find(key) != s._set->end()) {
    s._set->erase(key);
  }
  pthread_mutex_unlock(&s.lock);
}

#endif  // LABEL_SET_H_
