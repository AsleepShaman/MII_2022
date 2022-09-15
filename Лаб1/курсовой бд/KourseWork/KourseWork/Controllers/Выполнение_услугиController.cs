using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using KourseWork.Models;

namespace KoursWork.Controllers
{
    public class Выполнение_услугиController : Controller
    {
        private AudioStudioDBEntities2 db = new AudioStudioDBEntities2();

        // GET: Выполнение_услуги
        public ActionResult Index()
        {
            var выполнение_услуги = db.Выполнение_услуги.Include(в => в.Договоры).Include(в => в.Перечень_услуг).Include(в => в.Песни);
            return View(выполнение_услуги.ToList());
        }
        public ActionResult Error()
        {
            return View();
        }

        // GET: Выполнение_услуги/Details/5
        public ActionResult Details(int? id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Выполнение_услуги выполнение_услуги = db.Выполнение_услуги.Find(id);
            if (выполнение_услуги == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(выполнение_услуги);
        }

        // GET: Выполнение_услуги/Create
        public ActionResult Create()
        {
            ViewBag.Номер_договора = new SelectList(db.Договоры, "Номер_договора", "Паспорт_заказчика");
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование");
            ViewBag.Номер_песни = new SelectList(db.Песни, "Номер_песни", "Название_песни");
            return View();
        }

        // POST: Выполнение_услуги/Create
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "Номер_услуги,Срок_выполнения,Номер_песни,Номер_договора,Код_услуги,Стоимость")] Выполнение_услуги выполнение_услуги)
        {
            if (ModelState.IsValid)
            {
                try
                {
                    db.Выполнение_услуги.Add(выполнение_услуги);
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Error!";
                    return RedirectToAction("Error");
                }
            }

            ViewBag.Номер_договора = new SelectList(db.Договоры, "Номер_договора", "Паспорт_заказчика", выполнение_услуги.Номер_договора);
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование", выполнение_услуги.Код_услуги);
            ViewBag.Номер_песни = new SelectList(db.Песни, "Номер_песни", "Название_песни", выполнение_услуги.Номер_песни);
            return View(выполнение_услуги);
        }

        // GET: Выполнение_услуги/Edit/5
        public ActionResult Edit(int? id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Выполнение_услуги выполнение_услуги = db.Выполнение_услуги.Find(id);
            if (выполнение_услуги == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            ViewBag.Номер_договора = new SelectList(db.Договоры, "Номер_договора", "Паспорт_заказчика", выполнение_услуги.Номер_договора);
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование", выполнение_услуги.Код_услуги);
            ViewBag.Номер_песни = new SelectList(db.Песни, "Номер_песни", "Название_песни", выполнение_услуги.Номер_песни);
            return View(выполнение_услуги);
        }

        // POST: Выполнение_услуги/Edit/5
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "Номер_услуги,Срок_выполнения,Номер_песни,Номер_договора,Код_услуги,Стоимость")] Выполнение_услуги выполнение_услуги)
        {
            if (ModelState.IsValid)
            {
                try{db.Entry(выполнение_услуги).State = EntityState.Modified;
                db.SaveChanges();
                return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка редактирования!";
                    return RedirectToAction("Error");
                }
            }
            ViewBag.Номер_договора = new SelectList(db.Договоры, "Номер_договора", "Паспорт_заказчика", выполнение_услуги.Номер_договора);
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование", выполнение_услуги.Код_услуги);
            ViewBag.Номер_песни = new SelectList(db.Песни, "Номер_песни", "Название_песни", выполнение_услуги.Номер_песни);
            return View(выполнение_услуги);
        }

        // GET: Выполнение_услуги/Delete/5
        public ActionResult Delete(int? id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Выполнение_услуги выполнение_услуги = db.Выполнение_услуги.Find(id);
            if (выполнение_услуги == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(выполнение_услуги);
        }

        // POST: Выполнение_услуги/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(int id)
        {
            try{Выполнение_услуги выполнение_услуги = db.Выполнение_услуги.Find(id);
            db.Выполнение_услуги.Remove(выполнение_услуги);
            db.SaveChanges();
            return RedirectToAction("Index");
            }
            catch
            {
                ViewBag.Data = "Ошибка удаления! Возможно, эти данные где-то есть...";
                return RedirectToAction("Error");
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                db.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
